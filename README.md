# Bahraini-Food-Lab

An AI-powered food recognition app designed to help tourists explore Bahrain's culinary culture. Visitors can upload an image of a meal, and the computer vision model detects and identifies recognizable Bahraini and Arabic dishes.

## Classes 
Asida, Falafel, Halwa, kabsah, luqaimat, Machboos, MandI, Mashawi, Nakhaj, Sambusa, Thareed

## Project Structure

| File | Purpose |
|---|---|
| `notebook/Bahraini_Food.ipynb` | Notebook for preparing the dataset, training the YOLO model, and evaluating its performance. |
| `streamlit/BH_food_detection.py` | Streamlit application that allows users to upload food images and view detection results. |
| `models/best.pt` | Trained YOLO model weights used for food detection. |
| `requirements.txt` | Python dependencies required to run the application. |

## How It Works

1. Upload a food image through the Streamlit application.
2. The trained YOLO model analyzes the image and detects recognizable food items.
3. Bounding boxes are drawn around the detected dishes.
4. The application displays the detected food names and their confidence scores.

## Model Used

| Task | Model |
|---|---|
| Food Detection | YOLO26 Large (`yolo26l.pt`) |
| Trained Weights | `best.pt` |

The model was fine-tuned using a custom Arabic food image dataset from Roboflow.

## Model Evaluation

The model was evaluated on the validation dataset using the following metrics:

| Metric | Score |
|---|---:|
| Precision | 81.84% |
| Recall | 82.85% |
| mAP@50 | 87.09% |
| mAP@50-95 | 63.89% |

# URL 
https://bahraini-food-detector.streamlit.app

## Technologies

- Python
- YOLO (Ultralytics)
- PyTorch
- OpenCV
- Streamlit
- Roboflow
- Git LFS
