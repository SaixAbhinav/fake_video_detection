import streamlit as st

# ---------- CONFIG ----------
st.set_page_config(page_title="Fake Image Detection", page_icon="🧠", layout="centered")

# ---------- INITIAL SETUP ----------
if "page" not in st.session_state:
    st.session_state.page = "overview"

# ---------- HIDE SIDEBAR ----------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# ---------- OVERVIEW PAGE ----------
if st.session_state.page == "overview":
    st.title("🧠 Fake Image Detection using Deep Learning")
    st.markdown("### An MCA Minor Project by **Sai Abhinav** (VIPS Learning)")
    st.divider()

    st.header("📘 Project Overview")
    st.write("""
    This project uses Deep Learning to distinguish between **real human faces** and **AI-generated (fake)** ones.  
    It leverages a trained Convolutional Neural Network (CNN) built with PyTorch.
    """)

    st.image(
        "https://miro.medium.com/v2/resize:fit:1400/1*Uu3MbpxP0rLPkXGH6kW7_w.png",
        caption="Example: Real vs AI-generated Faces",
        use_container_width=True
    )

    st.header("🎯 Motivation")
    st.write("""
    - Deepfakes threaten digital trust and authenticity.  
    - AI-generated images can mislead users and spread misinformation.  
    - This project aims to **detect and flag AI-generated media** effectively.  
    """)

    st.header("⚙️ Methodology")
    st.markdown("""
    1. **Dataset:** Real & Fake face datasets  
    2. **Model:** CNN trained using PyTorch  
    3. **App:** Streamlit-based web interface  
    """)

    st.divider()
    st.success("Click below to start detecting fake images.")

    if st.button("➡️ Next: Go to Fake Image Detector"):
        st.session_state.page = "detector"
        st.rerun()


# ---------- DETECTOR PAGE ----------
elif st.session_state.page == "detector":
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image
    from fake_image_detector import FakeImageDetector

    st.title("🔍 Fake Image Detector")
    st.write("Upload an image to check if it’s **AI-generated (Fake)** or **Real**.")
    st.divider()

    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FakeImageDetector().to(device)
        model.load_state_dict(torch.load("C:/Users/saiab/Desktop/code/models/fake_image_detector.pth", map_location=device,weights_only=False))
        model.eval()
        return model, device

    model, device = load_model()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🧠 Analyzing Image..."):
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                prob = output.item()

            label = "🔴 FAKE IMAGE DETECTED" if prob > 0.5 else "🟢 REAL IMAGE DETECTED"
            confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100

        st.subheader(label)
        st.progress(int(confidence))
        st.write(f"**Confidence:** {confidence:.2f}%")

        if prob > 0.5:
            st.warning("⚠️ This image is likely AI-generated or manipulated.")
        else:
            st.success("✅ This image appears authentic and unaltered.")

    st.divider()
    if st.button("⬅️ Back to Overview"):
        st.session_state.page = "overview"
        st.rerun()
