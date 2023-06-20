import os
import streamlit as st
import openai
import email
import email.utils
# Configure OpenAI
openai.api_type = "azure"
openai.api_base = "https://generativetesing12.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "7b6053efd02247279c877b52fd78ff36"

# Streamlit app
st.title("Email Generator")

# Text input
email_prompt = st.text_area("Email Prompt:", value="Write a product launch email for new AI-powered headphones that are priced at $79.99 and available at Best Buy, Target and Amazon.com. The target audience is tech-savvy music lovers and the tone is friendly and exciting.\n\n1. What should be the subject line of the email?\n2. What should be the body of the email?")

# Generate email
if st.button("Generate Email"):
    response = openai.Completion.create(
        engine="maltext",
        prompt=email_prompt,
        temperature=1,
        max_tokens=350,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        stop=None
    )
    
    # Extract generated email from response
    generated_email = response.choices[0].text.strip()
    
    # Display generated email
    st.subheader("Generated Email:")
    st.write(generated_email)