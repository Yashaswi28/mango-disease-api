import streamlit as st
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor  # Use this instead of ViTImageProcessor

from PIL import Image

# ✅ Define the model path correctly
model_path = "C:/Users/suraj/vit_project/mango_leaf_disease_model.pth"  # Update if needed

# ✅ Load the ViT model architecture (ensure it matches your trained model)
class CustomViT(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomViT, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            ignore_mismatched_sizes=True
        )
        self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)  # ✅ Fix classifier layer

    def forward(self, x):
        return self.model(x)

# ✅ Initialize the model
model = CustomViT()

# ✅ Load trained model weights safely
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)  # ✅ Prevents weight mismatch issues
model.eval()  # Set to evaluation mode

# Load ViT image processor

processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")


# Define mango disease classes
classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 
           'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Streamlit UI setup
st.set_page_config(page_title="🍃 Mango Leaf Disease Detection 🥭", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>🍃 Mango Leaf Disease Detection 🥭</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📷 Upload a Mango Leaf Image for Disease Detection", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    # Predict disease
    with torch.no_grad():
        outputs = model(inputs["pixel_values"])
        predicted_class = classes[torch.argmax(outputs.logits).item()]

    # Show prediction result
    st.markdown(f"<h2 style='text-align: center; color: #d32f2f;'>Prediction: {predicted_class}</h2>", unsafe_allow_html=True)
