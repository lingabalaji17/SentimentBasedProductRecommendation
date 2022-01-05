
from flask import Flask, request, jsonify, render_template

import numpy as np
import pandas as pd

import re
import time

import nltk

import pickle

app = Flask(__name__)


class Recommendation():

    def __init__(self):
        self.model = pickle.load(open('model/LogisticRegression.pkl', 'rb'))
        self.vectorizer = pd.read_pickle('reviews_vectorizer.pkl')
        self.user_final_rating = pickle.load(open('user_final_rating.pkl','rb'))
        self.data = pd.read_csv("dataset/sample30.csv")

    def getRecommendationByUser(self, user):
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    def getSentimentRecommendations(self, user):
        recommendations = self.getRecommendationByUser(user)
        temp = self.data[self.data.name.isin(recommendations)]
        X = self.vectorizer.transform(temp["reviews_text"].values.astype(str))
        temp["prediction"]= self.model.predict(X)
        temp = temp[['name','prediction']]
        temp=temp.groupby('name').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=list(temp.sort_values('positive_percent', ascending=False)[0:5].index)
        return final_list
        
recommendation = Recommendation()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    #get user from the html form
    user= request.form['userId']
    #convert text to lowercase
    user=user.lower()
    #items = recommendation.getRecommendationByUser(user)
    items = recommendation.getSentimentRecommendations(user)
    print(len(items))
    print(items)
    return render_template("index.html", data=items)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port='5000')


