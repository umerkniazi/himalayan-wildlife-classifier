import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
from src.model import get_model

st.set_page_config(page_title="Himalayan Wildlife Classifier", layout="centered")

@st.cache_resource
def load_model():
    class_names = ['himalayan_brown_bear', 'markhor', 'snow_leopard']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(torch.load('models/himalayan_wildlife_classifier.pth', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, class_names, device

model, class_names, device = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Himalayan Wildlife Classifier")
st.write("Upload an image to classify the wildlife species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    confidence_score = confidence.item() * 100
    
    if confidence_score < 80.0:
        label = "Unknown Object"
        color = "red"
    else:
        label = class_names[predicted_idx.item()].replace('_', ' ').title()
        color = "#00cc44"
        
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    width, height = annotated_image.size
    
    line_width = max(3, int(min(width, height) * 0.02))
    draw.rectangle([line_width, line_width, width-line_width, height-line_width], outline=color, width=line_width)
    
    st.image(annotated_image, use_container_width=True)
    
    st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.progress(confidence_score / 100.0)
    st.write(f"Confidence: **{confidence_score:.2f}%**")