import langid

def detect_language(text: str) -> str:
    lang, _ = langid.classify(text)
    return lang
