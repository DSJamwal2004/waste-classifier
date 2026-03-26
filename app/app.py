import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import get_model

# Page config
st.set_page_config(page_title="Waste Classifier", layout="centered")

# Title + Description
st.title("♻️ Waste Classification App")
st.markdown("Upload an image to classify waste into recyclable categories.")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Class descriptions
descriptions = {
    "cardboard": "Recyclable packaging material",
    "glass": "Reusable and recyclable glass items",
    "metal": "Metal waste like cans",
    "paper": "Paper waste products",
    "plastic": "Plastic containers and packaging",
    "trash": "Non-recyclable waste"
}

@st.cache_resource
def load_model():
    import gdown
    
    MODEL_PATH = "best_model.pth"

    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1edbiRNohL9rggZt6aWNyMUobueMIk6pZ"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = get_model(len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    predicted_class = classes[pred.item()]
    confidence_score = confidence.item()

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence_score:.2f}")

    st.write("📌 Description:")
    st.write(descriptions[predicted_class])

st.markdown("---")
st.caption("Built using PyTorch + ResNet18")