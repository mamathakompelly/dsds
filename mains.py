import streamlit as st
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Install required packages
#pip_packages = ['diffusers', 'accelerate', 'safetensors', 'transformers']
#for package in pip_packages:
 #   st.run(f"pip install {package}")

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

image = download_image(url)

prompt = "turn him into cyborg"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images

# Streamlit app code
st.title("Instruct-Pix2Pix Image Transformation")
st.write("Input Image:")
st.image(image, caption="Input Image", use_column_width=True)
st.write("Output Image:")
st.image(images[0], caption="Output Image", use_column_width=True)