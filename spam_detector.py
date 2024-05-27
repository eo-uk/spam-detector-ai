import string
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


class SpamDetector:

    def __init__(self):
        vectorizer = CountVectorizer(preprocessor=self._preprocess_email)
        self.model = make_pipeline(vectorizer, MultinomialNB())

    def _preprocess_email(self, email):
        email = email.lower()
        email = email.strip()
        email = email.translate(str.maketrans('', '', string.punctuation))
        words = email.split()
        return ' '.join(words)
    
    def train(self, data, labels):
        self.model.fit(data, labels)

    def is_spam(self, email):
        preprocessed_email = self._preprocess_email(email)
        result = self.model.predict([preprocessed_email])
        return bool(result[0])

    def save(self, path="model.pkl"):
        with open(path, 'wb') as file:
            pickle.dump(self.model, file) 

    def load(self, path="model.pkl"):
        with open(path, 'rb') as file:
            self.model = pickle.load(file)