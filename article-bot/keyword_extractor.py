from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Explicitly request CPU usage
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
model = KeyBERT(model=sentence_model)

def extract_keywords(text: str, top_n=5) -> list:
    keywords = model.extract_keywords(text, top_n=top_n, stop_words="english")
    return [kw[0] for kw in keywords]