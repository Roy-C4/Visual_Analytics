#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import plotly.express as px
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import dateutil.parser as parser
from datetime import datetime
from rake_nltk import Rake
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


# In[2]:


def prepare_data():
    # Import data
    df = pd.read_csv('email headers.csv', sep=",")
    # Fix columns
    df['From'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'])
    # df['Sender'] = [x.split('@')[0] for x in df['From']]
    # df['Sender'] = [x.replace('.', ', ') for x in df['Sender']]
 
    return df

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    words = ' '.join(words)
    return words

def all_text(in_dir):
    '''
    Loops over all articles and outputs a dictionary with the file name as key, 
    and as value a tuple with the date and the tokenized text.
    @in_dir: the directory of the articles
    '''
    alltext = {}
    date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|([A-Z][a-z]{2}\s\d{1,2},?\s\d{2,4})|(\d{4}[/-]\d{2}[/-]\d{2}|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b)"

    rake_nltk_var = Rake()

    for file_name in os.listdir(in_dir):
        if file_name.endswith('.txt'):
            input_path  = os.path.join(in_dir, file_name)
            with open(input_path, 'r') as infile:
                text = infile.read()
                # Find all matches of the date pattern in the text
                results = re.findall(date_pattern, text)
                for i in results:
                    for j in i:
                        if j != '':
                            article_date = j
                            article_date = parser.parse(article_date)
                            article_date = article_date.strftime("%Y/%m/%d")
                cleaned_text = clean_text(text)
                # rake_nltk_var.extract_keywords_from_text(cleaned_text)
                # keyword_extracted = rake_nltk_var.get_ranked_phrases()
                alltext[file_name] = [article_date, cleaned_text]

    return alltext



def find_keywords(all_text):
    pass


# In[3]:


in_dir = "C:/Users/didov/Desktop/DS&AI/Q3_VisualAnalytics/Visual_Analytics/Disappearance at GAStech/data/articles"
alltext = all_text(in_dir)


# In[24]:


sentence = alltext['0.txt'][1]
# lemmatizer = WordNetLemmatizer()
# out = " ".join([lemmatizer.lemmatize(wd) for wd in sentence.split()])
nodate={}
dates={}
for key, value in alltext.items():
#     for i in value[1].split():
#         if i == 'aardgasbedrijf':
#             print(key)
    nodate[key] = value[1]
    dates[key] = value[0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(list(nodate.values()))
word_list = vectorizer.get_feature_names_out()

df = pd.DataFrame(X.todense(), index=nodate.keys(), columns=vectorizer.get_feature_names_out())
df = df.sort_values(by='Date')

df
# Added [0] here to get a 1d-array for iteration by the zip function. 
# count_list = np.asarray(X.sum(axis=0))[0]
# word_counts = dict(zip(word_list, count_list))
# l = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
# print(l) 


# In[18]:


df['Date'] = dates
df[['kronos', 'Date']]


# In[23]:


def plot_timeline(dataframe, keyword):
    keyword = keyword.lower()
    df = dataframe[[keyword, 'Date']]
    fig = px.line(df, x='Date', y=keyword, title=f'Occurance of {keyword}')
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
    
plot_timeline(df, 'death')


# In[ ]:




