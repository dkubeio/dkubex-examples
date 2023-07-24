from transformers import pipeline
import json
from ray import serve
import os


@serve.deployment()
class BioGpt:
    def __init__(self):
        self.pipe_biogpt = pipeline("text-generation", model=os.environ["MODEL_PATH"])

    async def __call__(self, starlette_request):
        request = await starlette_request.body()
        data = json.loads(request)
        prompt = data["prompt"]
        max_length = data.get("max_length", 100)
        num_return_sequences = data.get("num_sequences", 5)
        output_biogpt = self.pipe_biogpt(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
        )
        result = output_biogpt[0]["generated_text"]
        return result

# 2: Deploy the deployment.
deploy = BioGpt.bind()

