import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


def preprocess_text(text: str) -> str:
    # remove html tags
    text = re.sub(r"<[^>]*>", " ", text)
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^a-zA-Z0-9äöüÄÖÜß ]+", " ", text)

    # remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("german")]
    text = " ".join(tokens)
    text = text.lower().strip()

    return text