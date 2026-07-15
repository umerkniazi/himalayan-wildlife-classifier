# Himalayan Wildlife Classifier

A computer vision project built with PyTorch and Streamlit to classify images of three endangered Himalayan species: the Snow Leopard, Markhor and Himalayan Brown Bear. The model uses transfer learning with ResNet18 and includes confidence thresholding to flag uncertain predictions instead of forcing a classification.

## Dataset Attribution

This project utilizes the **Wildlife Animals (Snowleopard, Brownbear, Markhor)** dataset.

* **Dataset Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/hunzaikashif49/wildlife-animals-snowleopard-brwonbear-markhor)
* **Description:** A curated collection of wildlife images representing three highly endangered, high-altitude species native to the Gilgit-Baltistan region.

## Architecture & Approach

* **Framework:** PyTorch
* **Model Architecture:** ResNet18 (Transfer Learning)
* **Training Strategy:** Feature Extraction (early layers frozen, custom 3-class fully connected head trained).
* **Data Augmentation:** RandomResizedCrop and RandomHorizontalFlip to prevent overfitting on a limited dataset.
* **UI/Frontend:** Streamlit for local, interactive inference.

## Performance

The model achieved **98.16% Validation Accuracy** across 5 epochs.

| Metric | Train | Validation |
| :--- | :--- | :--- |
| **Accuracy** | 91.77% | 98.16% |
| **Loss** | 0.2253 | 0.0631 |

## QA Testing, Edge Cases, & Limitations

To evaluate robustness, the model was tested against Out-of-Distribution (OOD) data and visually similar species. This exposed several known limitations inherent to standard image classification models:

1. **The "Softmax Trap" (OOD Data):** When fed an image outside the training domain, such as a household toaster, the model forced a classification due to the closed-world assumption of the Softmax function.
   * **Solution Implemented:** Added an 80% confidence threshold. The toaster scored ~55% and was correctly intercepted and flagged as an **Unknown Object**.

2. **Semantic Similarity & Background Bias:** When tested on a **Polar Bear**, the model confidently predicted "Himalayan Brown Bear" (93%). The network detected shared visual features such as body structure and snowy environments, resulting in a confident false positive.

3. **Fine-Grained Classification Limits:** When tested on an **Ibex**, the model predicted "Markhor" (98%). Because both animals share the *Caprinae* subfamily (similar coats, hooves, horns, and rocky habitats), the model lacked the fine-grained training data required to reliably differentiate between them.

**Future Improvements:** Transitioning from an Image Classifier (ResNet) to an Object Detector (YOLO), collecting more diverse wildlife images, and adding an explicit "Other Background" class to reduce bias.

## Local Setup & Usage

### 1. Install Dependencies

    pip install torch torchvision pillow streamlit

### 2. Run the Web Interface

To launch the interactive dashboard locally:

    streamlit run app.py

Upload any image via the browser UI to see the model's prediction and confidence score.