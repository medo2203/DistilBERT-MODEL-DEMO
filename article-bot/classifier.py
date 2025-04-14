# classifier.py
from transformers import pipeline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli",
                      device=-1,
                      use_fast=False)

def classify(text: str, lang: str) -> str:
    candidate_labels = {
        "en": ["politics", "technology", "health", "economy", "sports", "entertainment", "culture", "social media"],
        "fr": ["politique", "technologie", "santé", "économie", "sport", "divertissement", "culture", "réseaux sociaux"]
    }
    labels = candidate_labels.get(lang, candidate_labels["en"])
    result = classifier(text, labels)
    return result["labels"][0]
