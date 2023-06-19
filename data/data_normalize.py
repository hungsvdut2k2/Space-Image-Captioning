import re


def text_normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub("\s+", " ", text)
    text = text.strip()
    text = "<start> " + text + " <end>"

    return text
