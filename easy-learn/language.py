import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from string import punctuation
from nltk.corpus import stopwords


class NaiveBayesNGram():
    def __init__(
            self,
            n: int = 1,
            use_tfidf: bool = True,
            remove_stops: bool = False,
            remove_punctuation: bool = True,
            use_range: bool = True
        ):
        self.n = n
        self.use_tfidf = use_tfidf
        self.remove_stops = remove_stops
        self.remove_punctuation = remove_punctuation
        self.use_range = use_range
        self.bow_transformer = None
        self.model = None

    def _preprocess_text(self, text: str) -> str:
        stops = stopwords.words('english')
        if self.remove_punctuation:
            text = ''.join([char.lower() for char in text if char not in punctuation])
        if self.remove_stops:
            text = ' '.join([word for word in text.split() if word not in stops])
        return text

    def fit(self, documents, labels) -> None:
        if not type(documents) == pd.Series:
            raise NotImplementedError('Implement handling of non pandas format')
        documents = documents.apply(self._preprocess_text).tolist()
        if self.use_range:
            self.bow_transformer = CountVectorizer(ngram_range=(1,self.n)).fit(documents)
        else:
            self.bow_transformer = CountVectorizer(ngram_range=(self.n,self.n)).fit(documents)
        bow = self.bow_transformer.transform(documents)
        if self.use_tfidf:
            tfidf_transformer = TfidfTransformer().fit(bow)
            bow = tfidf_transformer.transform(bow)
        self.model = MultinomialNB().fit(bow,labels)

    def predict(self, documents) -> pd.Series:
        if not type(documents) == pd.Series:
            raise NotImplementedError('Implement handling of non pandas format')
        documents = documents.apply(self._preprocess_text).tolist()
        test_bow = self.bow_transformer.transform(preprocessed_text)

        


def main():

    train = pd.read_json('data/music_reviews_train.json',lines=True)[['reviewText','sentiment']].dropna()

    #print(data.head())
    clf = NaiveBayesNGram(remove_stops=True,remove_punctuation=True,use_tfidf=True)
    #clf.fit(train['reviewText'],train['sentiment'])
    #print(type(train['reviewText']))

if __name__ == '__main__':
    main()