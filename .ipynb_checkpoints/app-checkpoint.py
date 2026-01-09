import streamlit as st
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import timm
import io

# Define the model class (copied from your training code)
class LightHybridNet(nn.Module):
    def __init__(self, num_classes=8, dropout=0.5):
        super().__init__()
        self.mobilenet = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=False, num_classes=0)
        self.effnet = timm.create_model('efficientnet_b1', pretrained=False, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(1280 + 1280, num_classes)  # 2560 → 8

    def forward(self, x):
        f1 = self.mobilenet(x)
        f2 = self.effnet(x)
        f = torch.cat([f1, f2], dim=1)
        f = self.dropout(f)
        return self.head(f)

# Validation transforms (from your code)
val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Labels and human-readable mappings
LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
LABEL_MAP = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration',
    'H': 'Hypertension',
    'M': 'Myopia',
    'O': 'Other diseases/abnormalities'
}

# Smart prediction function (from your code)
def get_smart_preds(probs, thresh=0.5):
    preds = np.zeros_like(probs)
    for i, p in enumerate(probs):
        if p[0] > 0.8:  # Very confident Normal
            preds[i, 0] = 1
        else:
            mask = p > thresh
            if mask.sum() == 0:
                preds[i, np.argmax(p)] = 1
            else:
                preds[i] = mask.astype(int)
    return preds

# Load the model
@st.cache_resource  # Caches the model for faster reloads
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LightHybridNet().to(device)
    model.load_state_dict(torch.load('best_lighthybrid_odir_new.pth', map_location=device))
    model.eval()  # Set to inference mode
    return model, device

# Main app
st.title("Ocular Disease Classifier")
st.write("Upload a fundus image to classify for diseases. Model: LightHybridNet trained on ODIR-5K.")

model, device = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = np.array(img.convert('RGB'))
    transformed = val_transforms(image=img)
    img_tensor = transformed['image'].unsqueeze(0).to(device)  # Add batch dim

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # Probabilities

    # Get predictions
    preds = get_smart_preds(probs.reshape(1, -1))[0]  # Reshape for compatibility

    # Display results
    st.subheader("Prediction Results")
    st.write("Probabilities (0-1) and detected labels (thresholded):")
    for i, label in enumerate(LABELS):
        prob = probs[i]
        detected = "✅ Detected" if preds[i] == 1 else "❌ Not Detected"
        full_label = LABEL_MAP[label]
        st.write(f"**{full_label} ({label})**: Probability = {prob:.4f} | {detected}")

    if np.sum(preds) == 0:
        st.warning("No diseases detected (confident 'Normal').")