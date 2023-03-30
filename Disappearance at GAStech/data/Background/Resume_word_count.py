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
            stop_words = set(nltk.corpus.stopwords.words('english'))
            filtered_corpus = [[word for word in doc if word not in stop_words] for doc in tokenized_corpus]

            cv = CountVectorizer()
            X = cv.fit_transform([' '.join(doc) for doc in filtered_corpus])
            words = cv.get_feature_names()
            word_counts = cv.transform([text])
            
            for i in range(len(words)):
                word_dict[words[i]] = word_counts.toarray()[0][i]

sorted_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}



print(sorted_dict)
                

            