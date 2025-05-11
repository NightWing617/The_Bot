
# news_sentiment_scraper.py

import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load HuggingFace sentiment pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def fetch_news_sentiment(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [tag.get_text() for tag in soup.find_all("h2")]  # Adjust tag as needed
    sentiments = []

    for headline in headlines:
        sentiment = sentiment_analyzer(headline)[0]
        sentiments.append({
            "headline": headline,
            "label": sentiment['label'],
            "score": round(sentiment['score'], 4)
        })
    
    return sentiments
