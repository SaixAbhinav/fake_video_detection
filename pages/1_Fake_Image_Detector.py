import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ------------------------------------------------------------
# 🧩 Define the Model
# ------------------------------------------------------------
import torch
import torch.nn as nn

class FakeImageDetector(nn.Module):
    def __init__(self):
        super(FakeImageDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



# ------------------------------------------------------------
# ⚙️ Load Model Safely (cached in Streamlit)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FakeImageDetector().to(device)

    # Load checkpoint safely (ignore mismatched shapes)
    state_dict = torch.load("C:/Users/saiab/Desktop/code/models/fake_image_detector.pth", map_location=device)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model, device


model, device = load_model()

# ------------------------------------------------------------
# 🧾 Preprocessing Pipeline
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # must match training size
    transforms.ToTensor(),
])

# ------------------------------------------------------------
# 🎨 Sidebar Information
# ------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Fake Image Detector")
    st.markdown("Detect whether an uploaded image is **Real** or **AI-generated (Fake)** using a CNN model.")
    st.divider()
    st.markdown("### 👨‍💻 Developer Info")
    st.write("**Name:** Sai Abhinav")
    st.write("**Course:** MCA, VIPS Learning")
    st.write("**Year:** 2025–26")
    st.write("**Tech Stack:** PyTorch • Streamlit • CNN")
    st.divider()
    st.caption("Developed with ❤️ using Streamlit & PyTorch")

# ------------------------------------------------------------
# 🌐 Main App Section
# ------------------------------------------------------------
st.title("🕵️‍♂️ Fake Image Detector")
st.write("Upload an image to check if it is **AI-generated** or **Real**.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            confidence = output.item()

        # Label and confidence
        if confidence > 0.5:
            label = "🔴 FAKE IMAGE DETECTED"
            confidence_score = confidence * 100
        else:
            label = "🟢 REAL IMAGE DETECTED"
            confidence_score = (1 - confidence) * 100

    # --------------------------------------------------------
    # 📊 Display Results
    # --------------------------------------------------------
    st.subheader(label)
    st.progress(int(confidence_score))
    st.write(f"**Confidence:** {confidence_score:.2f}%")

    if confidence > 0.5:
        st.warning("⚠️ This image is likely AI-generated or manipulated.")
    else:
        st.success("✅ This image appears authentic and unaltered.")

    # --------------------------------------------------------
    # 💾 Downloadable Result (UTF-8 safe)
    # --------------------------------------------------------
    with open("prediction.txt", "w", encoding="utf-8") as f:
        f.write(f"{label}\nConfidence: {confidence_score:.2f}%")

    with open("prediction.txt", "rb") as f:
        st.download_button("📄 Download Result", f, file_name="result.txt")

# ------------------------------------------------------------
# 🧾 Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 | Fake Image Detector by Sai Abhinav | MCA Minor Project")

st.divider()
if st.button("🔄 Restart App"):
    st.rerun()
