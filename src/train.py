import warnings
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model

warnings.filterwarnings("ignore", category=UserWarning)


def train_model():
    data_dir = 'data/splits'

    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)

    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training on device: {device}")

    model = get_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=0.001
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    num_epochs = 20

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(
                        outputs,
                        labels
                    )

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += (
                    loss.item() * inputs.size(0)
                )

                running_corrects += torch.sum(
                    preds == labels.data
                )

            epoch_loss = (
                running_loss / dataset_sizes[phase]
            )

            epoch_acc = (
                running_corrects.double()
                / dataset_sizes[phase]
            )

            print(
                f"{phase.capitalize()} "
                f"Loss: {epoch_loss:.4f} "
                f"Acc: {epoch_acc:.4f}"
            )

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                os.makedirs(
                    'models',
                    exist_ok=True
                )

                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'class_names': class_names
                    },
                    'models/himalayan_wildlife_classifier.pth'
                )

                print(
                    f"Saved best model "
                    f"(accuracy: {best_acc:.4f})"
                )

        scheduler.step()

    print("\nTraining complete.")
    print(
        f"Best validation accuracy: {best_acc:.4f}"
    )


if __name__ == '__main__':
    train_model()