import nltk
import string
from nltk.corpus import stopwords

# these downloads are no-ops after the first run, nltk checks a local cache
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
PUNCTUATIONS = set(string.punctuation)


def transform_text(text):
    # lowercase everything so "FREE" and "free" are treated the same
    text = text.lower()

    # split into words (this also breaks up things like "cannot" -> "can", "not")
    tokens = nltk.word_tokenize(text)

    # drop punctuation/numbers-only junk and common stopwords like "the", "is", "and"
    filtered_tokens = [
        word for word in tokens
        if word.isalnum() and word not in STOP_WORDS
    ]

    return " ".join(filtered_tokens)
