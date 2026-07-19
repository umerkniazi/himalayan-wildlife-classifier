import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import get_model

st.set_page_config(
    page_title="Himalayan Wildlife Classifier",
    layout="centered"
)


@st.cache_resource
def load_model():
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "models/himalayan_wildlife_classifier.pth",
        map_location=device
    )

    class_names = checkpoint["class_names"]

    model = get_model(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, class_names, device


model, class_names, device = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

st.title("Himalayan Wildlife Classifier")
st.write(
    "Upload an image to identify whether it contains a "
    "Himalayan brown bear, markhor, or snow leopard. "
    "Images of other species or unsupported subjects "
    "will be reported as 'Not a Supported Species'."
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        use_container_width=True
    )

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    confidence_score = confidence.item() * 100
    predicted_class = class_names[predicted_idx.item()]

    if confidence_score < 80 or predicted_class == "other":
        st.warning(
            f"Prediction: Not a Supported Species "
            f"({confidence_score:.2f}%)"
        )
    else:
        st.success(
            f"Prediction: "
            f"{predicted_class.replace('_', ' ').title()} "
            f"({confidence_score:.2f}%)"
        )

    st.progress(confidence_score / 100)

    with st.expander("View class probabilities"):
        probs = probabilities[0].cpu().numpy()

        results = sorted(
            zip(class_names, probs),
            key=lambda x: x[1],
            reverse=True
        )

        for name, prob in results:
            st.write(
                f"**{name.replace('_', ' ').title()}**: "
                f"{prob * 100:.2f}%"
            )