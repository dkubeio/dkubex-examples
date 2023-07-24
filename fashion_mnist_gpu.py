import mlflow.pytorch
from ray import serve
import os
import logging

from io import BytesIO
from PIL import Image
from starlette.requests import Request
from typing import Dict

import torch
from torchvision import transforms




@serve.deployment(ray_actor_options={"num_gpus":1})
class ImageModel:
    def __init__(self):
        self.logger = logging.getLogger("ray.serve")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"devices: {self.device}")
        self.model = mlflow.pytorch.load_model(os.environ["MODEL_PATH"], map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.preprocessor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        

    async def __call__(self, starlette_request: Request) -> Dict:
        image_payload_bytes = await starlette_request.body()
        pil_image = Image.open(BytesIO(image_payload_bytes))
        self.logger.info("[1/3] Parsed image data: {}".format(pil_image))

        pil_images = [pil_image]  # Our current batch size is one
        input_tensor = torch.cat(
            [self.preprocessor(i).unsqueeze(0) for i in pil_images]
        ).to(self.device)
        self.logger.info("[2/3] Images transformed, tensor shape {}".format(input_tensor.shape))

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        self.logger.info("[3/3] Inference done!")
        return {"class_index": int(torch.argmax(output_tensor[0]))}


deploy = ImageModel.bind()
