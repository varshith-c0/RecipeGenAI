from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from generate_recipe import generate_recipe
from core.yt_utils import _extract_yt_videoID, _get_yt_video_metadata, extract_recipe_cnt_from_yt_url
from core.tracing import setup_tracing, trace_function, add_span_attribute

app = Flask(__name__)

# Setup tracing
setup_tracing()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/process_recipe", methods=["POST"])
@trace_function("process_recipe")
def process_recipe():
    inp_j = request.get_json()
    recipe_cnt = inp_j.get('recipeText', '')
    yt_link = inp_j.get('ytVideoLink', '')
    is_ing_check_enabled = inp_j.get('ingredientCheck', True)
    is_instr_check_enabled = inp_j.get('instructionCheck', True)

    add_span_attribute("request.has_recipe_text", bool(recipe_cnt))
    add_span_attribute("request.has_yt_link", bool(yt_link))
    add_span_attribute("request.recipe_text", recipe_cnt)
    add_span_attribute("request.yt_link", yt_link)
    add_span_attribute("request.ingredient_check_enabled", is_ing_check_enabled)
    add_span_attribute("request.instruction_check_enabled", is_instr_check_enabled)

    if not recipe_cnt and not yt_link:
        add_span_attribute("validation.input_provided", not bool(recipe_cnt) and not bool(yt_link))
        return jsonify({"status": False, "output": 'Enter valid input'})

    if yt_link:
        MAX_VID_DURATION = 30 * 60.0 # 30 minutes
        videoID = _extract_yt_videoID(yt_link)
        if not videoID:
            add_span_attribute("validation.yt_link_valid", False)
            return jsonify({"status": False, "output": 'Incorrect URL.'})

        metadata = _get_yt_video_metadata(videoID)
        if not metadata:
            add_span_attribute("validation.is_metadata_present", False)
            return jsonify({"status": False, "output": "Invalid URL."})
        if metadata['duration_s'] > MAX_VID_DURATION:
            return jsonify({"status": False, "output": f"Can only process videos with duration less than {int(MAX_VID_DURATION)} minutes."})

        is_valid, recipe_cnt, reason = extract_recipe_cnt_from_yt_url(yt_link, metadata)
        add_span_attribute("yt_video_processsing_result.success", is_valid)
        add_span_attribute("yt_video_processsing_result.extracted_recipe_cnt", recipe_cnt)

        if not is_valid:
            add_span_attribute("yt_video_processsing_result.failure_reason", reason)
            return jsonify({"status": is_valid, "output": reason})

    is_valid, output = generate_recipe(
        recipe_cnt, is_ing_check_enabled, is_instr_check_enabled
    )

    yt_recipe = recipe_cnt if yt_link else None

    add_span_attribute("output.success", is_valid)
    add_span_attribute("output.output", output)
    return jsonify({"status": is_valid, "yt_recipe": yt_recipe, "output": output})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8100, debug=True)
