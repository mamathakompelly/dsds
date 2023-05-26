import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = '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'

# Function to generate images using DALL·E model
def generate_images(prompt, num_images=3):
    response = openai.Completion.create(
        engine='davinci',
        prompt=prompt,
        max_tokens=100,
        n=num_images,
        stop=None,
        temperature=0.7
    )

    image_urls = []

    for choice in response.choices:
        if 'image' in choice:
            image_urls.append(choice['image'])

    return image_urls

# Streamlit app
st.title("DALL·E Image Generator")

# Get user input prompt
prompt = st.text_area("Enter your prompt")

if st.button("Generate Images"):
    # Generate images
    image_urls = generate_images(prompt)

    # Display images
    for i, image_url in enumerate(image_urls):
        try:
            st.image(image_url, use_column_width=True, caption=f"Image {i+1}")
        except Exception as e:
            st.error(f"Error loading image {i+1}: {e}")
