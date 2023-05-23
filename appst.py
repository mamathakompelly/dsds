import streamlit as st
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Set up the Streamlit app
st.title("Instruct-Pix2Pix Image Transformation")
st.write("Upload an Image:")

# Function to download the image from URL
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Load the Pix2Pix model
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Function to process the uploaded image
def process_image(image, prompt):
    images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
    return images[0]

# File uploader to get the image from the user
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = PIL.Image.open(uploaded_file)
    image = image.convert("RGB")

    # Prompt for transformation
    prompt = st.text_input("Enter the prompt for transformation")

    if st.button("Transform"):
        # Perform the image transformation
        transformed_image = process_image(image, prompt)

        # Display the transformed image
        st.image(transformed_image, caption="Output Image", use_column_width=True)
