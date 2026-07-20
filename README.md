# Himalayan Wildlife Classifier

A computer vision project built with PyTorch and Streamlit to classify images of three endangered Himalayan species: the Snow Leopard, Markhor and Himalayan Brown Bear.

> **Note:** Inspired by research into AI-assisted camera traps used to monitor wildlife in northern Pakistan. This project explores how dataset quality, class design and evaluation affect the performance of real-world computer vision systems.

## Dataset

The model is trained on the [Wildlife Animals (Snowleopard, Brownbear, Markhor)](https://www.kaggle.com/datasets/hunzaikashif49/wildlife-animals-snowleopard-brwonbear-markhor) dataset, a collection of wildlife photographs representing three endangered species native to the Gilgit-Baltistan region.

To improve robustness, a manually curated **Other** class was added containing visually similar animals including domestic cats, leopards, polar bears, wolves and goats. This allows the model to reject unsupported species instead of forcing every image into one of the target classes.

## Methodology

The project uses transfer learning with a pretrained ResNet18 backbone and a custom four-class classification head.

Key design decisions include:

- Transfer learning instead of training from scratch to make better use of a relatively small dataset
- An internal **Other** class to reduce false positives on unsupported species
- An **80% confidence threshold** to reject uncertain predictions
- Data augmentation using `RandomResizedCrop` and `RandomHorizontalFlip`
- A Streamlit interface for local interactive inference

During inference, predictions assigned to the **Other** class or below the confidence threshold are displayed as **Not a Supported Species**.

## Results

The model achieved **96.55% validation accuracy**.

| Class | Precision | Recall | F1-score |
| :--- | ---: | ---: | ---: |
| Himalayan Brown Bear | 0.9412 | 0.9412 | 0.9412 |
| Markhor | 0.9796 | 0.9600 | 0.9697 |
| Other | 0.9518 | 0.9634 | 0.9576 |
| Snow Leopard | 1.0000 | 1.0000 | 1.0000 |

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

## Testing and Limitations

The model was evaluated on both the validation set and external images.

Adding the **Other** class and confidence threshold significantly reduced false positives, but visually similar animals can still be confused in some cases. For example, domestic cats may resemble snow leopards, domestic goats may resemble markhors and other bear species may resemble Himalayan brown bears.

The project also highlighted an important lesson: improving the dataset often has a greater impact than changing the model. The training data contains conventional wildlife photographs rather than camera trap images, so it does not fully represent the conditions the model would encounter in practice. A larger and more diverse dataset would likely improve performance more than experimenting with different architectures.

## Future Improvements

- Fine-tune the ResNet18 backbone
- Expand the **Other** class with additional species
- Collect more diverse wildlife imagery, particularly camera trap data if it becomes available
- Explore object detection models such as YOLO to detect and classify animals within an image

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload an image through the Streamlit interface to receive a prediction and confidence score.