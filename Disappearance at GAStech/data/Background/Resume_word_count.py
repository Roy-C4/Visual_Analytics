import os
import nltk
from contextlib import redirect_stdout
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


in_dir_path = r'D:\roy\selected\TU\books\visual_analytics\Disappearance at GAStech\data\Background\resumes\txt_versions'

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha()
             and word not in stop_words]
    return ' '.join(words)


word_dict = {}

for file_name in os.listdir(in_dir_path):
    if file_name.endswith('.txt'):
        input_path = os.path.join(in_dir_path, file_name)
        with open(input_path, 'r') as infile:
            text = infile.read()    
            tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in text]
            print(tokenized_corpus)s
            cleaned_text = clean_text(tokenized_corpus)

                

            