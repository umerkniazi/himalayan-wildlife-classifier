import os
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import get_model


def evaluate():
    data_dir = "data/splits"

    dataloaders, dataset_sizes, _ = get_dataloaders(data_dir)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "models/gb_wildlife_model.pth",
        map_location=device,
    )

    class_names = checkpoint["class_names"]

    model = get_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

    accuracy = (
        sum(p == t for p, t in zip(y_pred, y_true))
        / len(y_true)
    )

    print(f"\nValidation Accuracy: {accuracy:.4f}\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    plt.title("Confusion Matrix")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)

    output_path = "results/confusion_matrix.png"
    plt.savefig(output_path, dpi=300)

    print(f"Confusion matrix saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    evaluate()