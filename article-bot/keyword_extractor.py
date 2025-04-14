from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Use a French-compatible model
sentence_model = SentenceTransformer("dangvantuan/sentence-camembert-base", device="cpu")
model = KeyBERT(model=sentence_model)

def extract_keywords(text: str, top_n=5) -> list:
    keywords = model.extract_keywords(text, top_n=top_n, stop_words=None)
    return [kw[0] for kw in keywords]
