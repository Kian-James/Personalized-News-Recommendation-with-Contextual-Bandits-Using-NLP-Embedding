from sklearn.feature_extraction.text import TfidfVectorizer

class EmbeddingModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=300)

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()