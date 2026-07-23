# Nosh Recipe Generation AI — Architecture & Internals

This document explains the entire product: what it does, how every module works,
what every LLM tool and every deterministic validator does, and how a request
flows from a pasted recipe (or YouTube link) to a hardware-executable command
script. Read it top to bottom and you should understand everything happening
under the hood.

---

## Table of Contents

1. [What this product is](#1-what-this-product-is)
2. [The Nosh hardware model](#2-the-nosh-hardware-model)
3. [System architecture](#3-system-architecture)
4. [End-to-end request flow](#4-end-to-end-request-flow)
5. [Flask layer (`flask-app.py`)](#5-flask-layer)
6. [YouTube extraction (`core/yt_utils.py`)](#6-youtube-extraction)
7. [RAG grounding (`core/rag_tool.py`)](#7-rag-grounding)
8. [The orchestrator (`core/orchestrator.py`)](#8-the-orchestrator)
   - [Output schema](#81-output-schema)
   - [The four LLM tools](#82-the-four-llm-tools)
   - [The system instruction](#83-the-system-instruction)
   - [Run phases](#84-run-phases)
   - [Every deterministic validator](#85-every-deterministic-validator)
   - [Post-loop deterministic actions & ship guards](#86-post-loop-deterministic-actions--ship-guards)
9. [Command post-processor (`core/cmd_processor.py`)](#9-command-post-processor)
10. [Supporting modules](#10-supporting-modules)
11. [Hardware calibration tables (reference)](#11-hardware-calibration-tables)
12. [Interactive serving retry flow](#12-interactive-serving-retry-flow)
13. [Failure modes & guarantees](#13-failure-modes--guarantees)
14. [Running the service](#14-running-the-service)

---

## 1. What this product is

**Nosh** is an autonomous single-pot cooking robot. This service is its
recipe brain: it takes a free-form recipe — pasted text or a YouTube cooking
video — and converts it into a **validated, hardware-executable command
script**: which ingredients go in which tray, when each tray dispenses, how
much oil/water/spice to dispense, at what heat, with what stir/wait cadence,
for how long.

The hard part is not generating plausible commands — an LLM does that easily.
The hard part is generating commands that are **physically executable and
correctly proportioned** for a specific machine with strict limits (5 trays,
400 g effective per tray, 8 supported spices, a single pan, calibrated water
and salt tables). The architecture reflects this: **one LLM reasoning pass,
wrapped on every side by deterministic code** — deterministic tools feeding it
verified numbers, deterministic validators auditing its output, a
deterministic repair loop forcing it to fix its own mistakes, and deterministic
guards that refuse to ship anything physically impossible.

---

## 2. The Nosh hardware model

Everything downstream is shaped by what the machine can and cannot do:

| Capability | Details |
|---|---|
| Heating | Induction stove, integer heat levels (`stove heat 1..8`), `heat_till [°C]` |
| Stirring | One arm: `saute` (~16 s), `mix` (~30 s), `mix_liquid` (~45 s) per stir |
| Ingredient trays | **5 trays**, each dispensed **once**, all contents together |
| Spice dispenser | Exactly 8 spices: Salt, Turmeric, Chilli powder, Garam masala, Coriander powder, Cumin powder, Cumin seeds, Mustard seeds. Each dispense = ¼ tsp |
| Liquid dispensers | Auto water and oil (pourable oil only — never ghee/butter/solid fats) |
| AI-assisted cooking | Vision models that watch one ingredient cook: Onion, Tomato, Rice, Potato, Okra, Suji, Atta, Pasta, Millet (`cook X till cooked`) |
| Cannot | Knead, roll, ferment, bake, deep-fry, grill, steam under pressure, strain, transfer between pots |

**Tray physics.** A tray's practical limit depends on how the food packs, so
each ingredient's weight is multiplied by a *scale factor* by cut type, and the
sum ("effective weight") must stay ≤ **400 g** per tray:

| Tray class | Raw limit | Scale factor |
|---|---|---|
| `large_cut` | 200 g | ×2.0 |
| `small_cut` | 300 g | ×1.33 |
| `liquid` | 400 g | ×1.0 |
| `boneless_meat` | 400 g | ×1.0 |
| `bone_in_meat` | 300 g | ×1.33 |
| `grain` | 200 g | ×2.0 |
| *(unclassified)* | — | ×1.33 (conservative default) |

**Serving ceiling.** Salt counts and rice-water volumes are hardware-calibrated
tables keyed **1–4 servings**. There is no entry 5 — a bigger recipe must be
scaled down, never extrapolated.

---

## 3. System architecture

```
                        ┌─────────────────────────────────┐
                        │      Browser (templates/        │
                        │        index.html)              │
                        │  recipe text / YT link /        │
                        │  servings picker / retry banner │
                        └──────────────┬──────────────────┘
                                       │ POST /process_recipe (JSON)
                        ┌──────────────▼──────────────────┐
                        │   flask-app.py                  │
                        │   input validation, servings    │
                        │   validation, tracing spans     │
                        └───────┬───────────────┬─────────┘
                     YT link?   │               │  recipe text
                ┌───────────────▼──┐            │
                │ core/yt_utils.py │            │
                │ YouTube Data API │            │
                │ + Gemini video   │            │
                │ understanding    │            │
                └───────────────┬──┘            │
                                │ extracted recipe text
                        ┌───────▼───────────────▼─────────┐
                        │  generate_recipe.py             │
                        │  └ core/recipe_generator.py     │
                        │     (thin coordinator)          │
                        └──────────────┬──────────────────┘
                                       │
                 ┌─────────────────────▼───────────────────────┐
                 │        core/orchestrator.py  (the brain)    │
                 │                                             │
                 │  1. RAG retrieval (core/rag_tool.py→Qdrant) │
                 │  2. Gemini tool-calling loop (≤8 rounds)    │
                 │       tools: unit converter, serving        │
                 │       estimator, verified timing lookup,    │
                 │       AI-command validator                  │
                 │  3. Structured JSON extraction (Pydantic)   │
                 │  4. Deterministic validate→repair loop      │
                 │       (≤3 rounds, 14 validators)            │
                 │  5. Deterministic fixes & ship guards       │
                 │       (auto-split, overweight refusal,      │
                 │        serving suggestion)                  │
                 └─────────────────────┬───────────────────────┘
                                       │ OrchestratorOutput
                        ┌──────────────▼──────────────────┐
                        │  core/cmd_processor.py          │
                        │  deterministic translation to   │
                        │  hardware text: vegmap, salt    │
                        │  table, rice water table, spice │
                        │  sanitising, cook expansion     │
                        └──────────────┬──────────────────┘
                                       │ final command script (text)
                                       ▼
                              JSON response to browser
```

External services: **Gemini API** (LLM + embeddings + video understanding),
**YouTube Data API** (metadata), **Qdrant** (vector DB of hardware-verified
recipes), **Arize Phoenix** (OpenTelemetry trace collector).

---

## 4. End-to-end request flow

### Path A — pasted recipe text

1. Browser POSTs `{recipeText, ingredientCheck, instructionCheck, requestedServings}`.
2. `flask-app.py` validates input (non-empty, servings 1–4 or blank).
3. `generate_recipe()` → `run_recipe_generator()` → `run_orchestrator()`.
4. Orchestrator: RAG lookup → LLM tool loop → JSON extraction → repair loop →
   guards (details in §8.4).
5. On success, `to_distribution_extended()` converts the Pydantic output into
   the legacy `DistributionExtended` shape and `run_cmd_processor()` renders
   the final hardware text.
6. Response: `{status: true, output: "<command script>"}`.

### Path B — YouTube link

1. Browser POSTs `{ytVideoLink, ...}`.
2. `_extract_yt_videoID()` parses the URL (watch/shorts/embed/youtu.be forms).
3. `_get_yt_video_metadata()` (YouTube Data API) fetches title, description,
   duration. Videos over **30 minutes** are rejected up front.
4. `extract_recipe_cnt_from_yt_url()` sends the video URI + metadata prompt to
   Gemini's **video understanding** (`fps=1.5`, `MEDIA_RESOLUTION_LOW`, fixed
   seed) which returns a structured recipe text (name / ingredients with
   quantities / numbered steps). Quantities not stated in the video are
   estimated from visual cues.
5. The extracted text then follows Path A from step 3. The response includes
   `yt_recipe` (the extracted text) so the UI writes it into the textarea —
   any retry re-uses the text and **never re-processes the video**.

### Failure path — recipe too large (interactive retry)

See §12. Short version: the API returns
`{status: false, error_code: "too_large", max_feasible_serving: N, output: "<reason>"}`,
the UI shows a retry banner with the serving picker prefilled to N, and one
click resubmits with `requestedServings=N`.

---

## 5. Flask layer

`flask-app.py` — three routes:

| Route | Purpose |
|---|---|
| `GET /` | Serves `templates/index.html` |
| `POST /process_recipe` | The single product endpoint |

`process_recipe` responsibilities, in order:

1. Parse JSON: `recipeText`, `ytVideoLink`, `ingredientCheck` /
   `instructionCheck` (strictness toggles, default true), `requestedServings`.
2. **Servings validation**: blank/absent → `None` (auto); otherwise must parse
   as an int in 1–4, else an immediate clean failure message.
3. Reject empty input (`Enter valid input`).
4. YouTube branch (Path B above) with clean user-facing errors for bad URLs,
   missing metadata, over-long videos, and videos Gemini cannot access
   (private/restricted → 403/404 from the model is translated into a
   human-readable message, not a 500).
5. Call `generate_recipe(...)` with the strictness flags and `target_serving`.
6. **Structured failure passthrough**: when generation fails with a dict
   payload, `error_code` and `max_feasible_serving` are lifted into the JSON
   response next to the human-readable `output` string.
7. Tracing: every meaningful attribute (inputs, flags, results) is attached to
   the request span via `add_span_attribute`.

The frontend (`templates/index.html`) is a single Tailwind page: recipe
textarea, YT link input, two strictness checkboxes, a **Servings dropdown**
(Auto/1/2/3/4), an output pane, and a hidden amber **retry banner** that
appears on `too_large` failures.

---

## 6. YouTube extraction

`core/yt_utils.py`:

- `_extract_yt_videoID(url)` — pure URL parsing; supports `youtube.com/watch?v=`,
  `/shorts/`, `/embed/`, `/v/`, and `youtu.be/` forms.
- `_get_yt_video_metadata(videoID)` — YouTube Data API `videos().list`;
  returns `{title, description, duration_s}` or `None`.
- `_yt_link_2_recipe_prompt(metadata)` — builds the extraction prompt: asks for
  recipe name, complete quantified ingredient list, numbered steps; instructs
  the model to estimate unstated quantities from visual cues.
- `extract_recipe_cnt_from_yt_url(url, metadata)` — the Gemini call. The video
  is passed as a `file_data` part with `VideoMetadata(fps=1.5)` and
  `MEDIA_RESOLUTION_LOW` (cost/latency control), `seed=1` for stability.
  - `ClientError` 403/404 means *the model* can't ingest this specific video
    (private, age/region-restricted, removed) — returned as a clean user
    message, not an error.
- `_generate_with_retry(...)` — exponential backoff (2 s base, doubling,
  jitter, 5 attempts) on transient Gemini failures (408/429/5xx).

Model: `gemini-3.1-flash-lite` (same as the orchestrator).

---

## 7. RAG grounding

`core/rag_tool.py` — retrieves **hardware-verified reference recipes** from a
Qdrant collection (`recipes_v2`, ~610 documents; each document is a full dish
record with per-serving cook times, water volumes, oil quantities, and exact
ingredient weights validated on real Nosh hardware).

`get_rag_reference(query)` returns a dict used two ways:

1. **`context`** — the top `_CONTEXT_DOCS = 3` documents' full text, injected
   into the orchestrator prompt as "Similar Nosh-Tested Recipes". This grounds
   quantities, slot layouts, and step patterns.
2. **`cook_time_by_serving` / `water_by_serving` / `dish_name`** — parsed
   numeric tables from **one** chosen *timing reference* document, used later
   by deterministic validators as the authoritative target for total cook time
   and water volume.

**Timing-reference selection** (`_pick_timing_reference`): the top
`_TIMING_CANDIDATES = 5` hits are re-ranked by **ingredient coverage** — the
fraction of a candidate's own ingredients actually mentioned in the input
recipe. A lower-ranked candidate overrides rank 1 only when its coverage beats
it by a margin (`_COVERAGE_OVERRIDE_MARGIN = 0.15`) and it has at least
`_MIN_REF_INGREDIENTS = 2` recognizable ingredients. This is why a "Chicken
Donne Biryani" (coverage 0.33) can beat a semantically-closer-sounding
"Hyderabadi Chicken Biryani" (coverage 0.13) as the timing source.

Embeddings: `gemini-embedding-001`. The Qdrant client is a lazily-initialized
module singleton.

Helpers `pick_value_for_serving` / `pick_cook_time_for_serving` select the
right per-serving entry (exact match, else nearest) and report which serving
the value came from.

---

## 8. The orchestrator

`core/orchestrator.py` — the heart of the system. One Gemini model
(`_MODEL = "gemini-3.1-flash-lite"`, temperature 0) does all the reasoning; all
judgment about *whether the output is right* is done by code.

### 8.1 Output schema

`OrchestratorOutput` (Pydantic, enforced via Gemini's structured output):

- `is_recipe: bool` — false if the input isn't a single cookable recipe.
- `nosh_compatible: bool` — false if the machine can't execute it (with `reason`).
- `recipe_name`, `serving`, `course`, `cuisine`, `dish_type`,
  `consistency` (dry / semi gravy / gravy / whole meal), `pan_type`
  (source cookware, an oil/heat calibration signal only),
  `covered_cook_seconds` (how long the source recipe cooks lid-on — Nosh has
  no lid, so this feeds timing compensation).
- `prep_instructions: list[str]` — human prep the user does before loading
  trays (marinades, ground pastes, pressure-cooked components).
- `slots: list[_SlotOut]` — trays 1–5; each ingredient has `ingredient_name`,
  `quantity`, `unit`, `preparation_step` (canonical short prep, with one-word
  tags like `marinated` for shared preps), and `tray_class` (drives the weight
  scale factor).
- `updated_instructions` — the recipe steps rewritten to reference trays.
- `commands: list[str]` — the Nosh command script (pre-post-processing).

### 8.2 The four LLM tools

Declared in `_make_tools()`, dispatched by `_dispatch()`. All four are
**deterministic Python** — the LLM decides *when* to call them; the answers
come from code and data files, not model guesses.

**1. `convert_ingredients_to_grams`** → `core/tools.py:_convert_quantity_to_grams`
   Converts every ingredient to Nosh units: grams for solids, ml for
   water/oil, tsp for supported spices. Backed by
   `resources/ingredients_cleaned.xlsx` + `resources/mapping.json` (per-unit
   weight data per ingredient). Name matching: lemmatization + fuzzy
   `get_close_matches` (0.90 cutoff), with a **suffix-retry** so regional
   names ("seeraga samba rice", "kashmiri red chilli") borrow the weight of
   their head noun without losing their own name. For spices it converts tsp →
   **dispense count** (¼ tsp each, whole numbers, minimum 1) and returns the
   exact `N` for `spice X dispense N times` so the model never recomputes it.

**2. `estimate_serving_size`** → `core/tools.py:_estimate_serving_size`
   Called **only when the recipe doesn't state servings**. Finds the first
   *anchor ingredient* present (priority order: paneer 75 g/serving, chicken
   150, mutton 150, fish 150, prawns 120, egg 60, rice 100, dals 40,
   potato 100, pasta 85 — protein first, since protein quantity is the most
   reliable headcount signal in Indian home cooking) and returns
   `round(quantity / per_serving_g)` clamped to 1–4.

**3. `get_fallback_timing`** → `core/tools.py:_get_fallback_instructions`
   Looks up **hardware-verified cook sequences** for ~30+ known ingredients
   from `fallback_instr.py` (validated on real hardware). If a match exists,
   its durations/frequencies override anything the model would infer from the
   recipe or RAG. The match is plain English; the tool description teaches the
   exact translation rules into command syntax ("Cook N seconds while Xing
   every M" → `cook N seconds stir X every M`; bare "Cook N seconds" →
   `wait N seconds`). AI-assisted ingredients (onion, tomato, rice, …) ignore
   fallback matches and use `cook X till cooked` instead.

**4. `validate_nosh_commands`** → `_exec_validate_commands` →
   `core/ai_cmd_validator.py:ai_cmds_validator`
   Lets the model check its own draft commands for AI-cooking violations
   *before* finishing. Same engine the repair loop uses (§8.5, validator 4).

### 8.3 The system instruction

`_SYSTEM_INSTRUCTION` (~300 lines) teaches the model the entire hardware
contract: the Nosh overview and cannot-do list, the tray weight table and
distribution rules, the full command grammar, and several hard-won rules:

- **Never shrink one ingredient to fit a tray.** If the recipe doesn't fit,
  either scale the *whole* recipe (all quantities + serving together) or set
  `nosh_compatible=false`. Silently editing one quantity ships a dish that
  cooks half the food it claims.
- **Oil discipline**: recipes state deep-fry quantities; Nosh needs 10–15 ml.
  Solid fats (ghee/butter) go in trays, never the oil dispenser.
- **AI-cook pan-state rule**: `cook X till cooked` uses a vision model trained
  on X *alone* in the pan — invalid if a substantial earlier ingredient is
  already cooking (rice is the sole exception; same-tray companions and small
  aromatics don't count).
- **Rice water**: the system overrides the model's water amount with the
  calibrated table, so the model shouldn't agonize over it.
- **Onion/tomato rinse**: cmd_processor auto-appends `water dispense 50 ml` +
  `stir mix` after those AI steps — the model must not add them itself.
- Command bookends: `stove start` → `heat_till 75` first, `stove stop` last.

### 8.4 Run phases

`run_orchestrator(recipe_text, is_ing_check, is_instr_check, target_serving=None)`:

**Phase 1 — RAG retrieval** (pure code): `get_rag_reference(recipe_text)`;
context + timing tables logged in full for verifiability.

**Phase 2 — Tool-calling loop** (≤ `_MAX_TOOL_ROUNDS = 8` rounds): the prompt
(recipe + RAG block + optional *Required Serving Count* block when
`target_serving` is set + strictness mode) plus tools. Each round executes any
`function_call` parts and feeds results back. Loop ends when the model stops
calling tools.

**Phase 3 — Structured extraction**: one more call with
`response_schema=OrchestratorOutput` (JSON mode) → Pydantic-validated object.
If `is_recipe` or `nosh_compatible` is false here → clean failure with the
model's reason.

**Phase 4 — Deterministic validate → repair loop** (≤ `_MAX_REPAIR_ROUNDS = 3`):
every round runs *all* validators (§8.5) and sends every issue in **one**
batched repair message ("fix ALL of these; change nothing else") — fixing one
issue can shift another metric, so checking everything together avoids
fix-one-break-another ping-pong. Stopping conditions, in order:
1. no issues → success;
2. repair returned unparseable JSON → keep last good output;
3. issue count didn't strictly improve → stop early ("stuck");
4. round cap reached.
Remaining issues are logged loudly, and *most* still ship best-effort — except
the ones the guards below refuse.

**Phase 5 — Post-loop deterministic actions and guards** (§8.6).

API resilience: every Gemini call goes through `_generate_with_retry`
(exponential backoff 2 s/4 s/8 s/16 s/32 s + jitter on 408/429/5xx, 5 attempts).

### 8.5 Every deterministic validator

All live in `orchestrator.py`, all return plain-English issue strings consumed
by the repair loop, and none ask the LLM to judge anything — regex and
arithmetic only. In `_run_deterministic_validators` order:

**1. `_check_cook_time`** — computes total *active* time from the commands:
every `wait N` and `cook N`, every `stir` (16/30/45 s by type × count), plus
fixed estimates for AI steps (240 s for onion/tomato browning, 600 s for other
`till cooked`). Compares against the RAG timing reference's
`cook_time_by_serving` at this serving, tolerance ±18%
(`_COOK_TIME_TOLERANCE_RATIO`). Out of band → instructs the model to adjust
existing wait/cook/stir durations *proportionally* without touching
ingredients or structure.

**2. `_check_spice_commands`** — spice grammar police. Exact form
`spice [name] dispense [N] times` (trailing `times` REQUIRED — a malformed
line would otherwise be silently rewritten into a *different* spice by the
sanitizer's closest-match). Also rejects dispensing any spice outside the
supported 8 (those must go in a tray).

**3. `_check_tray_distribution`** — tray physics: ≤ 5 trays, numbers 1–5,
consecutive (no gaps); no water/oil/supported-spices inside trays; every tray
ingredient must be in **grams** (non-gram units make the weight check
meaningless, so they're flagged, not guessed); effective weight
(Σ quantity × scale factor) ≤ 400 g. The remedy text adapts: if all 5 trays
are in use it explicitly says splitting is impossible and names the only real
options (move to a tray with spare capacity at the same moment / scale the
whole recipe / declare incompatible) — because a repair loop given impossible
advice just thrashes.

**4. `_check_ai_commands`** — wraps `ai_cmds_validator` (§10): walks the
command list in order tracking pan contents; flags any `cook X till cooked`
that fires while a *different* substantial (>20 g) ingredient from an earlier
tray is in the pan. Rice exempt; same-tray companions exempt.

**5. `_check_cooked_rice_ai_command`** — fried-rice protection.
`cook rice till cooked` triggers the raw-rice program downstream (calibrated
700–1300 ml water + full cook cycle). If **no raw rice** exists in any tray —
every rice-named ingredient is `cooked/boiled/steamed/leftover` by name or
prep step, or is a rice product (flour, puffed, cake, paper) — the command is
flagged: remove it and heat the cooked rice through with `stir mix` + short
`wait`s instead.

**6. `_check_marinade_double_dispense`** — marinade coherence. If tray
ingredients carry a `marinated` prep tag, the marinade is already coating them
(applied in `prep_instructions`), so a *separate* tray ingredient named
`marinade`/`marination` double-counts every marinade component and pours raw
marinade into hot oil ahead of the food. Flagged with instructions to delete
the tray and its dispense. A standalone marinade tray with *nothing* tagged
marinated is left alone — dispensing a sauce over unmarinated food is a valid
design.

**7. `_check_cook_before_dispense`** — ordering: `cook X till cooked` must run
*after* the tray containing X has dispensed, or the robot watches an empty pan
for a full cycle. Matches head nouns ("rice" matches "seeraga samba rice").

**8. `_check_serving_range`** — serving ≤ 4 (the calibrated ceiling). Over →
scale the ENTIRE recipe down together; explicitly warns against just lowering
the number while leaving quantities alone (trays for 6, seasoning for 4).

**9. `_check_serving_consistency`** — anti-fudge check (skipped whenever the
range check fired, to avoid contradictory instructions in one round). Anchor
ingredients imply a headcount (chicken at 150 g/serving, etc.); if the implied
count differs from the stated serving by more than ×1.6, exactly one of the
two numbers is wrong — the issue tells the model to change ONE (restore the
shrunk quantity, or fix the serving count), and explicitly forbids rescaling
both together (which preserves the mismatch). Subtleties: potato is excluded
as an anchor (it's a co-star as often as a centrepiece); composite masses
(`marinated`, `ground to paste` tags) are skipped because their weight
includes the paste, not just the protein.

**10. `_check_dispense_coverage`** — set equality both ways: every declared
tray must have an `ingredient_tray N dispense` (else the user preps and loads
food that never enters the pan — the fix may be dispensing it, or moving a
garnish to `post_cooking_step`), and every dispense must reference a declared
tray.

**11. `_check_oil_water_commands`** — dispense sanity: exact grammar, numeric
positive amounts (cmd_processor parses positionally and would crash on
malformed lines); total oil ≤ 50 ml (deep-fry quantities copied from source
recipes are the classic offender); single water dispense ≤ 1500 ml
(pan overflow) — the water cap is skipped for rice dishes, where the amount is
replaced by the calibrated table anyway.

**12. `_check_water_volume`** — proportion, not just overflow: total dispensed
water vs the RAG reference's `water_by_serving` at this serving, tolerance
±25%. Counts the +50 ml auto-rinse cmd_processor adds after onion/tomato AI
steps (that water is invisible in the commands but real in the pan). Skipped
for rice (table override) and when the reference is under 50 ml (a splash, not
a steaming volume).

**13. `_check_spice_presence`** — completeness: any supported spice mentioned
in the recipe text (alias-aware; "mustard oil" is excluded as a false friend)
must actually be dispensed in the commands. Catches the "model forgot the
turmeric" class of bug.

**14. `_check_staged_ingredient_trays`** — staging: when the recipe text
staggers additions ("add X, cook, then add Y"), X and Y must not share a tray
(a tray dispenses everything at once). Tempering-scale riders ≤ 15 g are
allowed to share.

### 8.6 Post-loop deterministic actions & ship guards

After the repair loop, in order:

**a. Mid-repair verdict re-check.** A repair round may legitimately flip
`is_recipe`/`nosh_compatible` to false (the model realizes the recipe can't
fit). The pre-loop check can't see that, so it's re-checked here — otherwise a
self-declared-incompatible output (typically with emptied slots and zero
commands) would ship as a success. If the refusal is size-related (the last
output that still had slots contains overweight trays), the failure payload
carries `error_code: "too_large"` and a serving suggestion (§12).

**b. `_auto_split_overweight_trays`.** Deterministic last resort when a tray
is overweight and a tray slot is FREE: split it into two consecutive trays
dispensed at the same moment (identical contents, identical instant — the one
tray repair that cannot change how the dish cooks). Multi-ingredient trays
are greedily balanced heaviest-first; single oversized ingredients are
mass-halved. Tray numbers and every `ingredient_tray` command are renumbered
in step. Anything smarter (moving food between cooking *moments*) is
semantics-changing and stays the model's job.

**c. Final re-validation** — the loop only measures issues *before* each
repair, so the last repair's own result is measured here and logged. Cook-time
and water mismatches ship best-effort (logged, visible in telemetry).

**d. Overweight ship guard.** If any tray is *still* over 400 g effective, the
orchestrator **refuses to ship** — the hardware cup physically cannot hold it.
Returns a structured failure with per-tray detail, `error_code: "too_large"`,
and `max_feasible_serving` when computable.

**e. `_max_feasible_serving`.** Deterministic retry suggestion:
`floor(serving × capacity / total_effective_grams)` where capacity =
5 × 400 g × **0.85 packing efficiency** (raw capacity is unreachable — a 750 g
split across two trays fills neither; small items can't share trays across
moments), clamped to 1–4 and below the current serving. `None` if even one
serving can't fit. It's an estimate, not a proof — the retry re-runs the full
pipeline, which re-validates for real.

---

## 9. Command post-processor

`core/cmd_processor.py:run_cmd_processor(info_obj)` — pure deterministic
translation of the validated orchestrator output into the final hardware text.
Every transformation:

1. **Header block**: `#recipe name`, `#course`, `#consistency`, `#servings`.
2. **Preparation block**: `prep_instructions`, case-insensitively de-duplicated
   while preserving order.
3. **Slot listing**: one comment block per tray in *dispense order* (handles
   out-of-order dispensing), each ingredient as `name | qty | prep`. Display
   quantities round half-up with a floor of 1 g (`_fmt_qty`) — a real 0.4 g of
   cardamom must never print as 0; exact floats are still used internally.
4. **`vegmap`**: maps sanitized ingredient names (first ingredient of each
   tray, alphanumeric only) to dispense positions; duplicate names get numeric
   suffixes (`paneer`, `paneer2`).
5. **Command translation**, line by line:
   - `ingredient_tray N dispense` → `vegetable <mappedname> dispense`.
   - `heat_till T` → prefixed with `set_temperature_timeout` (60 s for ≤75 °C,
     120 s above) and, on the first heat, `stove heat 3`.
   - `stove level low/mid/high/very_high` → `stove heat 1/3/4/6`
     (`get_int_stove_heat`).
   - `oil/water dispense N ml` → amount snapped to the dispenser's step size
     (`process_oil_water_disp_cmd`: oil in 5 ml steps, water in 10 ml steps,
     round-half-up). A water dispense **> 250 ml** expands into a boil
     sequence: `stove heat 8` → dispense → `stir mix_liquid` →
     `set_temperature_timeout 120` + `heat_till 65` → restore previous heat.
   - `cook X till cooked`: onion is rewritten to `till golden_brown`; after
     onion/tomato the deglaze rinse (`water dispense 50 ml` + `stir mix 1
     times`) is appended automatically; sets flags for the rice/onion hooks
     below.
6. **`add_onion_specific_cmds`** — if no salt was dispensed before the onion
   browning step, inserts `spice salt dispense 1 times` (salt draws moisture
   and prevents burning).
7. **`add_rice_specific_cmds`** — the raw-rice program: rewrites the nearest
   preceding water dispense to the calibrated volume
   `{1: 700, 2: 900, 3: 1100, 4: 1300}` ml, keyed by **actual raw-rice grams
   in the trays** (100 g/serving; summed across split trays) rather than the
   serving label when the two disagree; marks post-rice tray dispenses
   `no_mix`; appends `exhaust off` + `wait 300 seconds`.
   **Skipped entirely when no raw rice is found in any tray** (pre-cooked-rice
   recipes) — the raw-rice detector is substring- and prep-aware
   (`cooked/boiled/steamed/leftover` in name or prep step excluded, rice
   flour/puffed rice/rice cake/rice paper excluded).
8. **`enclose_and_sanitize_spice_blocks`** — wraps every consecutive run of
   spice dispenses in `spice position 1 times` … `spice rest 1 times`
   (dispenser head movement), and sanitizes each line via
   `sanitize_spice_name`: grammar-parsed (never positional — a malformed line
   passes through untouched rather than being silently rewritten into a
   different spice), closest-match to the 8 canonical names, and fractional
   counts converted tsp→dispenses (×4, min 1).
9. **`fix_salt_dispense`** — total salt is entirely code-decided, from a
   hardware-calibrated table keyed by consistency × serving:
   | | 1 | 2 | 3 | 4 |
   |---|---|---|---|---|
   | dry | 1 | 2 | 3 | 4 |
   | whole meal | 2 | 4 | 5 | 5 |
   | gravy / semi-gravy | 2 | 3 | 4 | 5 |
   Deficits are distributed +1 across existing salt commands (remainder on the
   last); a *missing* salt dispense is inserted after the last tray dispense
   (an unsalted dish is a generation fault, not a preference). Serving is
   clamped to 1–4 with a loud warning (`_clamp_serv_size`) — damage limitation
   for a bug upstream validators should have caught.
10. **`process_cook_cmd`** — expands the combined grammar
    `cook [total] seconds|minutes stir [type] every [x] seconds|minutes` into
    explicit `stir`/`wait` pairs using per-stir durations
    (saute 16 s / mix 30 s / mix_liquid 45 s), normalizing minutes to seconds,
    merging adjacent stirs/waits, and folding sub-10 s remainders. A bare
    `cook N seconds` (invalid syntax) becomes `wait N seconds`. Malformed
    lines pass through with a warning instead of crashing.
11. **`post_cooking_step`** — appended at the end in `[...]` (serve/garnish
    instructions for the user).

---

## 10. Supporting modules

**`core/tools.py`** — the deterministic engines behind LLM tools 1–3 (§8.2).
Data: `resources/ingredients_cleaned.xlsx` (per-ingredient unit weights and
types) + `resources/mapping.json` (name → row mapping) + bundled NLTK wordnet
for lemmatization.

**`fallback_instr.py`** — dict of hardware-verified cook sequences per
ingredient (plain-English durations/frequencies validated on real hardware),
served by the `get_fallback_timing` tool.

**`core/ai_cmd_validator.py`** — the pan-state engine for AI-assisted cooking
(used by both the `validate_nosh_commands` tool and repair-loop validator 4):
walks commands in order, accumulates tray contents into a simulated pan,
flags `cook X till cooked` when a different ≥20 g ingredient from an earlier
tray is present. Rice exempt (`_MULTI_INGREDIENT_OK`); same-tray companions
exempt; lemmatized names ("lady finger" → "okra").

**`core/recipe_generator.py`** — thin coordinator: calls `run_orchestrator`,
maps `ServerError` → "Model overloaded", passes structured failure dicts
through, otherwise converts to `DistributionExtended`.

**`generate_recipe.py`** — top-level facade used by Flask: generation +
`run_cmd_processor`, returns `(bool, command_string | failure_dict)`.

**`core/utils.py`** — shared logger (`my_logger`): console + rotating file
`app.log` (~1 MB × 3 backups), DEBUG level when env `DEBUG=true`. Also
`func_timing_decorator` (logs every decorated function's wall time — the
`executed in N seconds` lines throughout the logs).

**`core/tracing.py`** — OpenTelemetry via Arize Phoenix (project
`nosh-recipe-gen`, gRPC collector). `setup_tracing()` at app start;
`@trace_function` wraps each pipeline stage in a span; `add_span_attribute`
attaches inputs/outputs/failures. Every request is fully traceable end-to-end.

**`core/distribute_ingredients.py` / `core/cmd_generator.py`** — legacy
modules from the previous multi-call architecture. Still imported **only** for
their Pydantic schema classes (`Slot`, `Ingredients`, `Consistency`,
`RecipeExtended`, `DistributionExtended`), which remain the interchange format
between orchestrator and cmd_processor.

**`providers/`** — dormant multi-provider LLM abstraction
(`gemini_llm.py`, `aws_bedrock_llm.py`, `open_router_llm.py`). Not referenced
by the current pipeline; kept intentionally so the product can switch LLM
providers later without re-writing the integration layer. Do not delete.

---

## 11. Hardware calibration tables

Quick reference for every calibrated constant in the system:

| Table | Values | Where |
|---|---|---|
| Tray scale factors | large_cut ×2.0, small_cut ×1.33, liquid ×1.0, boneless ×1.0, bone-in ×1.33, grain ×2.0, default ×1.33 | `orchestrator._TRAY_SCALE_FACTORS` |
| Tray effective limit | 400 g | `orchestrator._MAX_TRAY_EFFECTIVE_G` |
| Max trays | 5 | `orchestrator._MAX_TRAYS` |
| Max serving | 4 | `orchestrator._MAX_SERVING`, `cmd_processor._MAX_SERV_SIZE` |
| Rice water | 700/900/1100/1300 ml at 1–4 servings, keyed by rice mass at 100 g/serving | `cmd_processor.add_rice_specific_cmds` |
| Salt counts | consistency × serving table (§9.9) | `cmd_processor.fix_salt_dispense` |
| Serving anchors | paneer 75, chicken/mutton/fish 150, prawns 120, egg 60, rice 100, dals 40, potato 100, pasta 85 g/serving | `tools._ANCHOR_SERVING_GRAMS` |
| Stir durations | saute 16 s, mix 30 s, mix_liquid 45 s | `cmd_processor.process_cook_cmd`, mirrored in `orchestrator._STIR_SECONDS` |
| Cook-time tolerance | ±18% vs RAG reference | `orchestrator._COOK_TIME_TOLERANCE_RATIO` |
| Water tolerance | ±25% vs RAG reference | `orchestrator._WATER_TOLERANCE_RATIO` |
| Oil ceiling | 50 ml total | `orchestrator._MAX_TOTAL_OIL_ML` |
| Water overflow | 1500 ml per dispense | `orchestrator._MAX_WATER_ML` |
| Serving mismatch ratio | ×1.6 | `orchestrator._SERVING_MISMATCH_RATIO` |
| AI-cook browning/cook estimates | 240 s (onion/tomato) / 600 s (others) | `orchestrator._AI_BROWNING_SECONDS` / `_AI_COOKED_SECONDS` |
| Aromatics threshold (pan-state) | 20 g | `ai_cmd_validator._MIN_CONFUSING_INGREDIENT_G` |
| Minor co-tray riders (staging) | 15 g | `orchestrator._MINOR_CO_TRAY_G` |
| Packing efficiency (suggestion) | 0.85 | `orchestrator._max_feasible_serving` |
| Spice dispense unit | ¼ tsp | conversion in `_exec_convert_ingredients` |
| Dispenser step sizes | oil 5 ml, water 10 ml | `cmd_processor.process_oil_water_disp_cmd` |
| Max video length | 30 min | `flask-app.process_recipe` |

---

## 12. Interactive serving retry flow

Some recipes are simply too big for the machine (e.g. a 1 kg-chicken biryani:
chicken 1000 g effective + rice 1260 g effective vs 2000 g total capacity).
The system turns that dead end into a one-click recovery:

1. **Detection** — either the overweight ship guard fires, or the model
   refuses mid-repair while its last slotted output had overweight trays.
2. **Suggestion** — `_max_feasible_serving` computes the largest serving count
   that plausibly fits (§8.6e).
3. **Response** — `{status: false, error_code: "too_large",
   max_feasible_serving: N, output: "<human-readable reason>"}`.
4. **UI** — amber banner "Recipe too large for Nosh. Retry with N serving(s)?"
   + the Servings dropdown prefilled to N.
5. **Retry** — resubmits the recipe *text* (already in the textarea, even for
   YT flows — no video re-processing) with `requestedServings=N`.
6. **Forced scaling** — the prompt gains a *Required Serving Count* block: the
   model must scale every ingredient to exactly N servings and skip
   `estimate_serving_size`. The full validator suite then re-checks the scaled
   output like any other run.

The Servings dropdown also works standalone: a user can pick 1–4 up front and
never hit the failure at all. Blank = Auto (model decides, tools estimate).

---

## 13. Failure modes & guarantees

What the system **guarantees** ships or fails cleanly:

| Situation | Behavior |
|---|---|
| Input isn't a recipe | `is_recipe=false` → clean failure with model's reason |
| Machine can't cook it (kneading, deep-fry, multi-pot…) | `nosh_compatible=false` → clean failure with reason |
| Recipe too large for trays | **Never ships.** Structured `too_large` failure + serving suggestion (§12) |
| Model flips to incompatible mid-repair | Caught post-loop → clean failure (never ships empty slots as success) |
| Overweight tray with a free slot | Deterministically auto-split (same-moment, semantics-preserving) |
| `cook rice till cooked` on pre-cooked rice | Validator forces removal in repair; cmd_processor additionally never applies the raw-rice water program without raw rice in trays |
| Marinade dispensed separately while items are marked marinated | Validator forces removal in repair |
| Missing salt / forgotten spices | Salt inserted from calibrated table; spice presence validator flags omissions |
| Cook time / water off vs verified reference | Repaired if possible; otherwise ships best-effort with loud log + telemetry |
| Transient Gemini errors (429/5xx) | Exponential-backoff retries; exhausted → "Model overloaded. Please try again." |
| Private/restricted YouTube video | Clean user message, not a 500 |
| Serving outside 1–4 | Rejected at the API; internally clamped as last-resort damage limitation |

Design principle throughout: **the LLM proposes, code disposes.** Anything
calibrated (weights, water, salt, timing tolerances) lives in deterministic
tables; anything the LLM produces is audited by deterministic validators; and
anything physically impossible is refused rather than shipped.

---

## 14. Running the service

**Environment** (`.env`):

| Variable | Required | Used by |
|---|---|---|
| `GEMINI_API_KEY` | yes | orchestrator, embeddings, YT extraction |
| `YOUTUBE_DATA_API_KEY` | yes (asserted at import) | `yt_utils` metadata |
| `QDRANT_URI`, `QDRANT_API_KEY` | yes | RAG retrieval |
| `DEBUG` | no | `true` → DEBUG-level logging |

**Start**: `./run.sh` — activates `.venv`, then loops
`uv run gunicorn -b 0.0.0.0:8100 flask-app:app` (the `while true` wrapper
restarts gunicorn if it exits). Local dev: `python flask-app.py`
(Flask dev server, port 8100).

**Dependencies**: managed with `uv` (`pyproject.toml` + `uv.lock`).
Key: `google-genai` (LLM), `qdrant-client`, `flask` + `gunicorn`,
`google-api-python-client` (YouTube), `pandas`/`openpyxl` (ingredient data),
`nltk` (lemmatization), `arize-phoenix` + OpenTelemetry (tracing),
`pydantic` (schemas).

**Observability**: rotating `app.log` (full command scripts, validator
verdicts, repair rounds, timing per stage) + Phoenix traces (project
`nosh-recipe-gen`) with per-stage spans and attributes.

**Manual test corpus**: `test_recipes/test_suite.txt`.
