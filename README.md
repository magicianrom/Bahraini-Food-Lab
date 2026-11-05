# Bahraini-Food-Lab
An AI-powered food recognition app designed to help tourists explore Bahrain's culinary culture. 

## Project Structure

| File | Purpose |
|---|---|
| `notebook/Bahraini_Food.ipynb` | Evaluating pretrained Hugging Face models for restaurant review sentiment analysis. |
| `streamlit/BH_food_detection.py` | Streamlit application: accepts restaurant reviews and predicts sentiment and review category. |
| `requirements.txt` | Python dependencies required to run the application. |

## How It Works

1. Enter a restaurant review into the Streamlit application.

2. The sentiment analysis model classifies the review as positive, negative, or neutral.

3. The zero-shot classification model identifies the main topic of the review, such as Food Quality, Service, Price, or Cleanliness.

4. Both predictions and their confidence scores are displayed.

## Models Used

| Task | Hugging Face Model |
|---|---|
| Sentiment Analysis | `abhiprd20/nlp-sentiment-model` |
| Zero-Shot Classification | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |

## Technologies

- Python
- Hugging Face Transformers
- PyTorch
- Streamlit
- Git LFS

## Live Demo
https://restaurant-analyzer.streamlit.app
