import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import get_model


def predict(image_path, threshold=60.0):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        'models/gb_wildlife_model.pth',
        map_location=device
    )

    class_names = checkpoint['class_names']

    model = get_model(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )

    confidence_score = confidence.item() * 100

    predicted_class = class_names[
        predicted_idx.item()
    ]

    if confidence_score < threshold:
        predicted_class = "Unknown Object"

    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence_score:.2f}%")

    print("\nClass probabilities:")
    for name, prob in zip(
        class_names,
        probabilities[0]
    ):
        print(f"{name}: {prob.item() * 100:.2f}%")

    print()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(
            "Usage: python src/predict.py <path_to_image>"
        )
        sys.exit(1)

    predict(sys.argv[1])