from flask import Flask, render_template
app = Flask(__name__)
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from textblob import TextBlob

def clean_rating_count_column(rating):
    if isinstance(rating, str):
        rating = rating.replace(',', '')
        rating = float(rating)
        return rating
    return rating


def clean_rating_column(rating):
    if isinstance(rating, str):
        rating = rating.replace(',', '')
        rating = float(rating)
        return rating
    return rating


def clean_actual_price_column(price):
    if isinstance(price, str):
        price = price.replace(',', '').replace('₹','')
        price = float(price)
        return price
    return price

def clean_discounted_price_column(price):
    if isinstance(price, str):
        price = price.replace(',', '').replace('₹','')
        price = float(price)
        return price
    return price 

def clean_discounted_percentage(discount):
    if isinstance(discount, str):
        discount = discount.replace('%','')
        discount = float(discount)
        return discount
    return discount

def clean_rating_count(rating_count):
    if isinstance(rating_count, str):
        rating_count = rating_count.replace(',', '')
        rating_count = float(rating_count)
        return rating_count
    return rating_count

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

def get_subjectivity(text):
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    return subjectivity

def get_sentiment_name(sentiment):
    if sentiment > 0.5:
        return 'highly positive'
    elif sentiment > 0:
        return 'slightly positive'
    elif sentiment == 0:
        return 'neutral'
    elif sentiment < -0.5:
        return 'highly negative'
    else:
        return 'slightly negative'

pd.set_option('display.max_columns', None)
def load_data():
    df = pd.read_csv('datasets/amazon.csv')
    df ['clean_rating_count'] = df['rating_count'].apply(clean_rating_count_column)
    df ['clean_rating'] = df['rating_count'].apply(clean_rating_column)
    df['clean_discounted_price'] = df['discounted_price'].apply(clean_discounted_price_column)
    df['clean_actual_price'] = df['actual_price'].apply(clean_actual_price_column)
    df['clean_discount_percentage'] = df['discount_percentage'].apply(clean_discounted_percentage)
    df['clean_rating_count'] = df['rating_count'].apply(clean_rating_count)
    df['review_content']
    df['sentiment'] = df['review_content'].apply(get_sentiment)
    df['subjectivity'] = df['review_content'].apply(get_subjectivity)
    df['sentiment_name'] = df['sentiment'].apply(get_sentiment_name)
    df['clean_actual_price'].mean()
    df[df['clean_actual_price'] < 5000]
    df[df['sentiment'] < 0]
    df[df['sentiment'] < 0].reset_index(drop=True)
    
    return df



df = load_data()




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analysis/univariate')
def univariate():
    
    # clean_actual_price
    df.sort_values(by='clean_actual_price', ascending=False, inplace=True)
    # df.reset_index(inplace=True)
    fig1 = px.bar(df.head(100), y='clean_actual_price', color='clean_actual_price', height=800, hover_name='product_name')
    
    # histogram of clean_actual_price
    fig2 = px.histogram(df, x='clean_actual_price', nbins=100, height=500, log_y=True, title='Histogram of Actual Price')
    conclusion = 'Most of the products are priced between 0 and 2000'
    fig2.add_annotation(x=70000, y=2, text=conclusion, font_size=20)
    
    # bin products based on price in 10 bins
    bin_names = ['cheap', 'affordable', 'moderate', 'expensive', 'very expensive']
    df['price_bins'] = pd.cut(df['clean_actual_price'], bins=5, labels=bin_names)
    price_bins = df['price_bins'].value_counts()
    fig3 = px.bar(price_bins, y=price_bins, color=price_bins, title='Price Bins', log_y=True)
    
    # clean_rating_count
    df.sort_values(by='clean_rating_count', ascending=False, inplace=True)
    df.reset_index(inplace=True)
    fig4 = px.bar(df.head(100), y='clean_rating_count', color='clean_rating_count', height=800, hover_name='product_name')
    
    # histogram of clean_rating_count
    fig5 = px.histogram(df, x='clean_rating_count', nbins=1000, height=500, log_y=True, title='Histogram of clean_rating_count')
    conclusion = 'Most of the products are priced between 0 and 2000'
    fig5.add_annotation(x=70000, y=2, text=conclusion, font_size=20)
    
    return render_template('univariate.html', 
                           fig1=fig1.to_html(),
                           fig2=fig2.to_html(),
                           fig3=fig3.to_html(),
                           fig4=fig4.to_html(),
                           fig5=fig5.to_html())
  
@app.route('/analysis/bivariate')
def bivariate():
    
    px.scatter(df, x='clean_actual_price',
           y='clean_rating_count',
           color='clean_rating_count',
           hover_name='product_name',
           marginal_x='histogram', marginal_y='histogram')
    fig1 = px.scatter(df, x='clean_actual_price')
    
    # box plot
    px.box(df, x='clean_actual_price')
    fig2 = px.box(df, x='clean_actual_price')
    
    px.histogram(df, x='clean_rating')
    fig3=px.histogram(df, x='clean_rating')
    
    # correlation price and discount
    px.scatter(df, x='clean_actual_price', y='clean_discounted_price', color='clean_rating_count', hover_name='product_name')
    fig4=px.scatter(df, x='clean_actual_price', y='clean_discounted_price', color='clean_rating_count', hover_name='product_name')
    
    px.scatter(df, x='clean_actual_price', y='clean_discount_percentage', color='clean_rating_count', hover_name='product_name')
    fig5=px.scatter(df, x='clean_actual_price', y='clean_discount_percentage', color='clean_rating_count', hover_name='product_name')
    
    # 100 least rated products
    px.bar(df.tail(100), y='clean_rating_count', color='clean_rating_count', hover_name='product_name')
    fig6=px.bar(df.tail(100), y='clean_rating_count', color='clean_rating_count', hover_name='product_name')
    
    return render_template('bivariate.html',
                           fig1=fig1.to_html(),
                           fig2=fig2.to_html(),
                           fig3=fig3.to_html(),
                           fig4=fig4.to_html(),
                           fig5=fig5.to_html(),
                           fig6=fig6.to_html())
    
    
@app.route('/analysis/sentiment')
def sentiment():
    
    px.bar(df[df['sentiment'] < 0].reset_index(drop=True), y='sentiment',hover_name='product_name')
    fig1=px.bar(df[df['sentiment'] < 0].reset_index(drop=True), y='sentiment',hover_name='product_name')
    
    sentimentals = df['sentiment_name'].value_counts()
    px.pie(sentimentals, names=sentimentals.index, values=sentimentals, title='Sentiment Distribution')
    fig2=px.pie(sentimentals, names=sentimentals.index, values=sentimentals, title='Sentiment Distribution')
    
    return render_template('sentiment.html',
                           fig1=fig1.to_html(),
                           fig2=fig2.to_html())
    
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=False)
 

