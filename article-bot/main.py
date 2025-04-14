from utils import load_article
from keyword_extractor import extract_keywords
from classifier import classify
from language_detector import detect_language

article = load_article("sample.json")

text = f"{article['title']} {article['description']}"
language = detect_language(text)
keywords = extract_keywords(text)
category = classify(text, language)

print("\n📰 Article Analysis")
print("-" * 40)
print(f"Title: {article['title']}")
print(f"Detected Language: {language}")
print(f"Category: {category}")
print(f"Keywords: {keywords}")
