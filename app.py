
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
        self.vectorizer = pickle.load(open('reviews_vectorizer.pkl', 'rb'))
        self.user_final_rating = pickle.load(open('user_final_rating.pkl','rb'))
    

    def getRecommendationByUser(self, user):
         return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

recommendation = Recommendation()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def prediction():
    #get user from the html form
    user='aac06002' #request.form['userid']
    #convert text to lowercase
    user=user.lower()
    items = recommendation.getRecommendationByUser(user)
    print(len(items))
    return jsonify(user=user, data=items)


if __name__ == '__main__':
	app.run()


