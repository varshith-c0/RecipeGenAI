import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from core.utils import logger, agent_run_wrapper

load_dotenv()

_openRouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
assert _openRouter_api_key is not None, "Load OPEN_ROUTER_API_KEY in .env"

open_router_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=_openRouter_api_key,
)

class OpenRouterLLM:
    def __init__(self, provider: str, model_id: str):
        self.provider = provider
        self.model_id = model_id

    def run(
        self,
        input_txt: str,
        system_instruction: str,
        response_model: BaseModel,
        reasoning_effort: str = "low"
    ):
        # https://openrouter.ai/docs/use-cases/reasoning-tokens
        # https://openrouter.ai/docs/use-cases/reasoning-tokens?utm_source=chatgpt.com
        # https://platform.openai.com/docs/guides/structured-outputs
        if self.provider == "open_router_client":
            output = open_router_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user"  , "content": input_txt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_json_schema",
                        "schema": response_model.model_json_schema(),
                        "strict": True
                    }
                },
                extra_body={
                    "usage": {
                        "include": True
                    },
                    "reasoning": {
                        "effort": reasoning_effort
                    }
                },
            )
            msg = output.choices[0].message
            logger.info(f"open_router content: {msg.content}")
            logger.info(f"open_router thought: {msg.reasoning}")
            logger.info(f"stats: {getattr(output, "usage", None)}")
            return {
                "response": msg.content,
                "thought" : msg.reasoning,
                "stats"   : getattr(output, "usage", None)
            }

        if self.provider == "agno":
            agent = Agent(
                model=OpenRouter(
                    id=self.model_id,
                    api_key=_openRouter_api_key,
                    max_completion_tokens=10000,
                    reasoning_effort=reasoning_effort,
                    max_retries=1
                ),
                description=system_instruction,
                output_schema=response_model,
                use_json_mode=True,
                markdown=False,
                parse_response=True,
                reasoning=True
            )
            response = agent_run_wrapper(agent, input_txt)
            print(f"agno_open_router response: {response}")
            content = response.content
            return {
                "response": content if isinstance(content, str) else content.model_dump(),
                "thought" : response.reasoning_content
            }
