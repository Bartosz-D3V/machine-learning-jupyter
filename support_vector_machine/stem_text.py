from nltk.stem import PorterStemmer


def stem_text(text):
    ps = PorterStemmer()
    stemmed_text = []
    for word in text.split(" "):
        stemmed_text.append(ps.stem(word))
    return " ".join(stemmed_text)
