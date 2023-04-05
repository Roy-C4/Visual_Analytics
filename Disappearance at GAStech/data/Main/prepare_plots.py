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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def prepare_data():
    # Import data
    df = pd.read_csv('email headers.csv', sep=",")
    # Fix columns
    df['From'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'])

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

def vectorize_documents(article_dict):
    corpus = article_dict.values()
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(corpus)

    return X, vectorizer

def create_frequency_dataframe(only_output_dictionary=False):
    in_dir = "C:/Users/didov/Desktop/DS&AI/Q3_VisualAnalytics/Visual_Analytics/Disappearance at GAStech/data/data_prep/articles"
    alltext = all_text(in_dir)
    nodate={}
    dates={}

    for key, value in alltext.items():
        nodate[key] = value[1]
        dates[key] = value[0]
    
    if only_output_dictionary == False:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(list(nodate.values()))

        df = pd.DataFrame(X.todense(), index=nodate.keys(), columns=vectorizer.get_feature_names_out())
        df['Date'] = dates
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        return df
    else:
        return nodate
        


def plot_timeline(dataframe, keywords):
    keywords.append('Date')
    df = dataframe[keywords]
    df_long = df.melt(id_vars='Date', var_name='Word', value_name='Frequency')
    fig = px.histogram(df_long, x='Date', y='Frequency', title='Occurance of keywords', color='Word')
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()


def elbow_method(Y_sklearn):
    """
    This is the function used to get optimal number of clusters in order to feed to the k-means clustering algorithm.
    """

    number_clusters = range(1, 20)  # Range of possible clusters that can be generated
    kmeans = [KMeans(n_clusters=i, max_iter = 600) for i in number_clusters] # Getting no. of clusters 

    score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))] # Getting score corresponding to each cluster.
    score = [i*-1 for i in score] # Getting list of positive scores.
    
    plt.plot(number_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Method')
    plt.show()


def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, batch_size=1024, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    

def plot_tsne_pca(show_clusters=True):
    in_dir = "C:/Users/didov/Desktop/DS&AI/Q3_VisualAnalytics/Visual_Analytics/Disappearance at GAStech/data/data_prep/articles"
    alltext = all_text(in_dir)
    corpus = {}
    for key, value in alltext.items():
        corpus[key] = value[1]
    corpus = corpus.values()
    KM = KMeans(n_clusters=12, random_state=0, n_init="auto")
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(corpus)
    KM.fit(X)
    prediction = KM.predict(X)

    pca = PCA(n_components=2).fit_transform(np.asarray(X.todense()))
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(np.asarray(X.todense())))

    cluster_keywords = get_top_keywords(X, prediction, vectorizer.get_feature_names_out(), 5)
    new_labels = [cluster_keywords[i] for i in prediction]
   
    if show_clusters==False:
        fig = px.scatter(x=pca[:, 0], y=pca[:, 1], title="PCA Cluster plot")
    else:
        fig = px.scatter(x=pca[:, 0], y=pca[:, 1], color=new_labels, title="PCA Cluster plot")
    
    fig2 = px.scatter(x=tsne[:, 0], y=tsne[:, 1], color=new_labels, title="TSNE Cluster plot")
    
    return fig, fig2

    
def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    keyword_dict = {}
    for i,r in df.iterrows():
        keyword_dict[i] = ', '.join([labels[t] for t in np.argsort(r)[-n_terms:]])
        
    return keyword_dict

from sklearn.metrics.pairwise import cosine_similarity

def get_similar_articles(word):
    article_dict = create_frequency_dataframe(only_output_dictionary=True)
    X, vectorizer = vectorize_documents(article_dict)
    query_vector = vectorizer.transform([word])

    # Compute the cosine similarity between the query vector and all article vectors
    similarity_scores = cosine_similarity(X, query_vector)

    # Get the indices of the most similar articles
    most_similar_indices = similarity_scores.argsort(axis=0)[::-1][:10]

    article_titles = list(article_dict.keys())

    most_similar_articles = [(article_titles[int(i[0])], similarity_scores[i][0][0]) for i in most_similar_indices]

    # Print the titles of the most similar articles
    return most_similar_articles
