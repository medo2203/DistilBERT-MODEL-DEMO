import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from transformers import pipeline

# Force CPU usage
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=-1)  # -1 means CPU

def classify(text: str) -> str:
    candidate_labels = ["politics", "technology", "health", "economy", "sports", "entertainment", "culture", "social media"]
    result = classifier(text, candidate_labels)
    return result["labels"][0]  # return top category