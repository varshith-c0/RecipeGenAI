import re
import json
from dotenv import load_dotenv
from pydantic import BaseModel
import boto3

load_dotenv()

aws_bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2"
)

class AWSBedrockLLM:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def run(
        self,
        input_txt: str,
        system_instruction: str,
        response_model: BaseModel,
        reasoning_effort: str = "low"
    ):
        # https://aws.amazon.com/blogs/machine-learning/introducing-structured-output-for-custom-model-import-in-amazon-bedrock/
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html?utm_source=chatgpt.com
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": input_txt}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_json_schema",
                    "schema": response_model.model_json_schema(),
                    "strict": True
                }
            },
            "reasoning_effort": reasoning_effort,
            "stream": False
        }
        resp = aws_bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload)
        )
        body = json.loads(resp["body"].read().decode("utf-8"))
        output = body["choices"][0]["message"]["content"] # str
        print(f"aws_bedrock output: {output}")

        match = re.search(r"<reasoning>(.*?)</reasoning>(.*)", output, flags=re.DOTALL)
        if match:
            thought, response = match.group(1).strip(), match.group(2).strip()
        else:
            thought, response = None, output
        return {
            "response": process_aws_bedrock_output(response),
            "thought": thought
        }

def process_aws_bedrock_output(txt: str) -> str:
    """
    Output json by `openai.gpt-oss-120b-1:0` model starts
    with '{{' instead of '{'. This function extract proper
    output json.
    """
    key_idx = txt.find("\"is_recipe\"")
    # Scan backward to find the nearest '{'
    start = key_idx
    while start >= 0 and txt[start] != '{':
        start -= 1
    return txt if start < 0 else txt[start:]
