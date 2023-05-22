import io
from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

app = FastAPI()

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

def download_image(url):
  image = PIL.Image.open(requests.get(url, stream=True).raw)
  image = PIL.ImageOps.exif_transpose(image)
  image = image.convert("RGB")
  return image

@app.post("/process-image")
async def process_image(url: str, prompt: str):
  image = download_image(url)
  images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
  return {"result": images[0]}

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)