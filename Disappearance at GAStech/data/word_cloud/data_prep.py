import pandas as pd
import re
import plotly.express as px
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def prepare_data():
    # Import data
    df = pd.read_csv('email headers.csv', sep=",")
    print(df.head())
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
    return ' '.join(words)

def all_text(in_dir):
    '''
    Loops over all articles and outputs a dictionary with the file name as key, 
    and as value a tuple with the date and the tokenized text.
    @in_dir: the directory of the articles
    '''
    alltext = {}
    date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|([A-Z][a-z]{2}\s\d{1,2},?\s\d{2,4})|(\d{4}[/-]\d{2}[/-]\d{2}|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b)"


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
                cleaned_text = clean_text(text)
                alltext[file_name] = (article_date, cleaned_text)

    return alltext

in_dir = "D:/roy/selected/TU/books/visual_analytics/Disappearance at GAStech/data/articles"
alltext = all_text(in_dir)
#print(alltext['844.txt'])
#print(len(alltext.keys()))
