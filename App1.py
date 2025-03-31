import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import base64
import requests

# API Configuration
API_URL = "https://router.huggingface.co/hf-inference/models/impira/layoutlm-invoices"
headers = {"Authorization": "Bearer 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"}

client = InferenceClient(
    provider="hf-inference",
    api_key="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",
)

def query(image, question):
    payload = {
        "inputs": {
            "image": image,
            "question": question,
        },
    }
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload["inputs"]["image"] = img_str
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def process_image(image, question):
    result = query(image, question)
    if "error" in result:
        return f"Error: {result['error']}"
    
    return result[0]['answer']

# Streamlit UI
st.image("PragyanAI_Transperent.png")
st.title("Invoice Question Answering")
st.write("Upload an invoice image and ask a question.")

uploaded_file = st.file_uploader("Upload an invoice image", type=["png", "jpg", "jpeg"])
question = st.text_input("Enter your question")

if uploaded_file and question:
    image = Image.open(uploaded_file)
    
    with st.spinner("Processing..."):
        answer = process_image(image, question)
        
    st.write("**Answer:**", answer)
