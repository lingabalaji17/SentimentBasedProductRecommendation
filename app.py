from email import message
from flask import Flask, request, jsonify, render_template

import numpy as np
import pandas as pd

from model import SentimentModel

app = Flask(__name__)

sentiment_model = SentimentModel()


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
    items = sentiment_model.getSentimentRecommendations(user)

    #if(~(items is None)):
    return render_template("index.html", data=items)
    #else:
        #return render_template("index.html", data={message:"User Name doesn't exists, No product necommendations at this point of time!"})

@app.route('/predictSentiment', methods=['POST'])
def predict_sentiment():
    #get the review text from the html form
    review_text = request.form["review_text"]
    pred_sentiment = sentiment_model.classify_sentiment(review_text)
    return render_template("index.html", sentiment=pred_sentiment)


if __name__ == '__main__':
	app.run()


