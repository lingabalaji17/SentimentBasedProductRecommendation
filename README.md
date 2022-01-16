# Sentiment Based Product Recommendation

Problem Statement

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

 Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

 With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

Solution:

github link: https://github.com/lingabalaji17/SentimentBasedProductRecommendation

Application is live: https://senti-product-recommendation.herokuapp.com/

Data cleaning, Visualization, Text processing (NLP) and various Machine Learning models are applied on the review_text column to classify the User Sentiment (Sentiment Classification). Best Model is selected based on the various ML classification metrics (Accuracy, Precision, Recall, F1 Score, AUC). 
Colloborative Filtering Recommender system is created based on User-user and item-item approaches.RMSE evaluation metric is used for the evaluation.
Recommender system is filtered using the Sentiment classification model based on the predicted user sentiment of the reviews of the recommended products.
Machine Learning models are saved in the pickle files; Flask API is used to interface and test the Machine Learning models. Bootstrap and Flask jinja templates are used for setting up the User interface.
End to End application is deployed in Heroku 
