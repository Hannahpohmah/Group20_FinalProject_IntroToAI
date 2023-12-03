# Fake/Real Product Review Detection

# Problem statement
In the digital age of e-commerce, fake product reviews have become a widespread issue, impacting consumers and businesses. The problem lies in the prevalence of fraudulent reviews, both overly positive and excessively negative, distorting the authenticity of user-generated content and eroding trust in online marketplaces

## Overview

This project aims to develop a system that detects fake and real product reviews on eCommerce websites. Leveraging machine learning and natural language processing techniques, the goal is to create a model that can classify reviews accurately, helping users make informed decisions while shopping online.
The data set was made up of a massive collection of reviews having various distinct categories like Home and Office, Sports, etc. with each review having a corresponding rating, label i.e. CG(Computer Generated Review) and OR(Original Review generated by humans) and the review text.
If it is computer generated, it is considered fake otherwise not.
## Features

- *Data Collection:* Exploring and gathering a diverse dataset of product reviews from various eCommerce platforms.
- *Preprocessing:* Cleaning and preparing the textual data for analysis and model training.
- *Model Development:* Building NLP models LSTM, Bidirectional LSTM, pretrained Bert model to classify reviews as fake or real.
- *Evaluation:* Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
- *Deployment:* Deployed using Flask and HTML, hosted on Heroku, enabling users to access the review classification system online.
- 
1-BERT (Bidirectional Encoder Representations from Transformers)*
   - Description: Utilized pre-trained transformer-based models for contextual understanding of the reviews.
   - Performance: loss and accuracy on the test data included 0.4029478132724762, 0.8170974254608154 respectively
     
2. *Bidirectional Model*
   - Description: Leveraged bidirectional architectures to capture context from both directions within the reviews.
   - Performance: Loss on test data is: 0.2615833282470703
                  Accuracy on test data: 0.9048939347267151

3. *LSTM (Long Short-Term Memory) Model*
   - Description: Employed LSTM architecture to analyze the sequential nature of text data.
   - Performance: Achieved the best results among the three models, demonstrating superior performance in classifying fake and real product reviews.
     Loss on test data is: 0.2615833282470703
     Accuracy on test data: 0.9048939347267151
  

## Performance Evaluation

The LSTM model exhibited superior performance metrics in comparison to the BERT and Bidirectional models. Its ability to understand the sequential nature of reviews and capture contextual information contributed to its effectiveness in detecting fake and real product reviews.

Here is the link to my google collab file hosting the different models: https://colab.research.google.com/drive/1p6AQcHkqvX9_ueYYIvAHxtxulngFmC4u?usp=sharing 

## Deployment Details

This project has been deployed on Heroku, utilizing Flask for the backend and HTML for the frontend. Users can access the deployed application through the following link: https://hlstmtestapp-bf3cba8a0a69.herokuapp.com/

Watch a video to see how it works??? https://youtu.be/_KwUGj8VNYI

## Technologies Used

- Python
- Machine Learning Libraries (Scikit-learn, TensorFlow, Keras)
- Natural Language Processing (NLTK, spaCy)
- Web Scraping (Beautiful Soup, Scrapy)
- Flask (for building the API and backend)
- HTML/CSS (for the frontend)
- Heroku (for deployment)

  # Structure of files 
