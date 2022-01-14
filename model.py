import pickle
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

class SentimentModel():

    ROOT_PATH = "pickle/"
    MODEL_NAME ="sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"

    def __init__(self):
        self.model = pickle.load(open(SentimentModel.ROOT_PATH + SentimentModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(SentimentModel.ROOT_PATH + SentimentModel.VECTORIZER)
        self.user_final_rating = pickle.load(open(SentimentModel.ROOT_PATH + SentimentModel.RECOMMENDER,'rb'))
        self.data = pd.read_csv("dataset/sample30.csv")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def getRecommendationByUser(self, user):
        recommedations = []
        #try:
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
        #except:
           # print("No user exists")

    def getSentimentRecommendations(self, user):
        recommendations = self.getRecommendationByUser(user)
        #if(~(recommendations is None)):
        temp = self.data[self.data.id.isin(recommendations)]
        #temp["reviews_text_cleaned"] = temp["reviews_text"].apply(lambda x: self.preprocess_text(x))
        X = self.vectorizer.transform(temp["reviews_text"].values.astype(str))
        temp["prediction"]= self.model.predict(X)
        temp = temp[['name','prediction']]
        temp=temp.groupby('name').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=list(temp.sort_values('positive_percent', ascending=False)[0:5].index)
        return final_list

    def classify_sentiment(self, review_text):
        sentiment_text = ""
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        y_pred = self.model.predict(X)
        return y_pred

    def preprocess_text(self, text):

        #cleaning the review text (lower, removing punctuation, numericals, whitespaces)
        text = text.lower().strip()
        text = re.sub("\[\s*\w*\s*\]", "", text)
        dictionary = "abc".maketrans('', '', string.punctuation)
        text = text.translate(dictionary)
        text = re.sub("\S*\d\S*", "", text)

        #remove stop-words and convert it to lemma
        text = self.lemma_text(text)
        return text
    

    # This is a helper function to map NTLK position tags
    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def remove_stopword(self, text):
        words = [word for word in text.split() if word.isalpha() and word not in self.stop_words]
        return " ".join(words)

    def lemma_text(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(self.remove_stopword(text))) # Get position tags
        # Map the position tag and lemmatize the word/token
        words =[self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] 
        return " ".join(words)