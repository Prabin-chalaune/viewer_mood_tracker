import re
import requests
import string
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from urllib.parse import urlparse

nltk.download('wordnet')

ps = PorterStemmer()

stopwords=set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
                'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
                'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                'couldn', 'didn', 'doesn','hadn', 'hasn','haven','isn', 'ma', 'mightn', 'mustn',
                'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'\d+','',text)
    return text
def rem_wspaces(text):
    return ' '.join(text.split())
def rem_tags_urls(text):
    text = re.sub(r"http?://\S+|www.\.\S+", "", text)
    return re.sub(r'<.*?>','',text)
def tokenize(text):
    text = text.strip('.')
    return text.split()
def rem_stopwords(text):
    t_text = tokenize(text)
    f_text = [word for word in t_text if word not in stopwords]
    return f_text
def text_preprocess(text):
    #Remove puctuations, numbers and convert string into lower case
    text = clean_text(text)
    #Remove HTML tags and URLs
    text = rem_tags_urls(text)
    #Remove whitespaces
    text = rem_wspaces(text)
    #Tokenize the text
    text = rem_stopwords(text)
    return text
def stem_text(text):
    word_tokens = tokenize(text)
    return [ps.stem(words) for words in word_tokens]

def text_processing_pipeline(text):
    text = text_preprocess(text)
    text = ' '.join(text)
    text = stem_text(text)
    text = ' '.join(text)
    return text
