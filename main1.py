import streamlit as st
from PIL import Image
import requests
import io
import torch
import PIL.ImageOps

from transformers import AutoModel

model_id = "timbrooks/instruct-pix2pix"
model = AutoModel.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def download_image(url):
  image = Image.open(requests.get(url, stream=True).raw)
  image = PIL.ImageOps.exif_transpose(image)
  image = image.convert("RGB")
  return image

def transform_image(image):
  # Preprocess the image
  preprocess = st.experimental_wrapped_callback(model.preprocess_input)
  inputs = preprocess(image=image, return_tensors="pt")
  inputs = inputs.to(device)

  # Perform the image transformation
  outputs = model(inputs)
  transformed_image = outputs.pixel_values[0].cpu().numpy()

  # Postprocess the transformed image
  postprocess = st.experimental_wrapped_callback(model.postprocess_output)
  transformed_image = postprocess(pixel_values=transformed_image)

  return transformed_image

# Streamlit app code
st.title("Instruct-Pix2Pix Image Transformation")
st.write("Input Image:")
file = st.sidebar.file_uploader("Upload a file")
image_bytes = file.read()
image = Image.open(io.BytesIO(image_bytes))
image = image.convert("RGB")
# image = PIL.ImageOps.exif_transpose(uploaded_file)
st.image(image, caption="Input Image", use_column_width=True)

if st.button("Transform"):
  # Perform the image transformation
  transformed_image = transform_image(image)

  # Display the output image
  st.write("Output Image:")
  st.image(transformed_image, caption="Output Image", use_column_width=True)
