# GB Wildlife Classifier

A custom computer vision pipeline built with PyTorch and Streamlit to classify camera-trap images of wildlife native to Gilgit-Baltistan: the Snow Leopard, Markhor, and Himalayan Brown Bear.

## Dataset Attribution

This project utilizes the **Wildlife Animals (Snowleopard, Brownbear, Markhor)** dataset.

* **Dataset Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/hunzaikashif49/wildlife-animals-snowleopard-brwonbear-markhor)
* **Description:** A curated collection of camera-trap and field photography tracking three highly endangered, high-altitude species native to the Gilgit-Baltistan region.

## Architecture & Approach

* **Framework:** PyTorch
* **Model Architecture:** ResNet18 (Transfer Learning)
* **Training Strategy:** Feature Extraction (early layers frozen, custom 3-class fully connected head trained).
* **Data Augmentation:** `RandomResizedCrop` and `RandomHorizontalFlip` to prevent overfitting on a limited dataset.
* **UI/Frontend:** Streamlit for local, interactive inference.

## Performance

The model achieved **98.16% Validation Accuracy** across 5 epochs.

| Metric | Train | Validation |
| :--- | :--- | :--- |
| **Accuracy** | 91.77% | 98.16% |
| **Loss** | 0.2253 | 0.0631 |

## QA Testing, Edge Cases, & Limitations

To ensure robustness, the model was stress-tested against Out-of-Distribution (OOD) data and visually similar species. This exposed several known limitations inherent to standard Image Classification models:

1. **The "Softmax Trap" (OOD Data):** When fed an image of a household toaster, the model forced a classification due to the closed-world assumption of the Softmax function. 
   * *Solution Implemented:* Added an 80% confidence threshold. The toaster scored ~55% and was correctly intercepted and flagged as an **Unknown Object**.
2. **Semantic Similarity & Background Bias:** When tested on a **Polar Bear**, the model confidently predicted "Himalayan Brown Bear" (93%). The network successfully detected the structural shape of a bear and combined it with the snowy background (a valid context in the dataset), resulting in a highly confident false positive.
3. **Fine-Grained Classification Limits:** When tested on an **Ibex**, the model predicted "Markhor" (98%). Because both animals share the *Caprinae* subfamily (tan coats, hooves, horns, rocky terrain), the model lacked the granular training data required to differentiate scimitar horns from corkscrew horns.

*Future Improvements:* Transitioning from an Image Classifier (ResNet) to an Object Detector (YOLO) and adding an explicit "Other Background" class to mitigate bias.

## Local Setup & Usage

### 1. Install Dependencies
```bash
pip install torch torchvision pillow streamlit
```

### 2. Run the Web Interface
To launch the interactive dashboard locally:
```bash
streamlit run app.py
```
Upload any image via the browser UI to see the model's prediction and confidence score.