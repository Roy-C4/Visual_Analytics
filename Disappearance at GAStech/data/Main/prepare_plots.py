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
from sklearn.metrics.pairwise import cosine_similarity


def prepare_data():
    '''
    Imports the email headers csv file. 
    Outputs a dataframe.
    '''

    # Import data
    df = pd.read_csv('email headers.csv', sep=",")
    # Fix columns
    df['From'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'])

    return df

def clean_text(text):
    '''
    Strips and Tokenizes an article.
    Outputs the tokenized text as a list of words.
    '''

    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    words = ' '.join(words)
    return words

def all_text(in_dir):
    '''
    Loops over all articles and outputs a dictionary with the file name as key, 
    and as value a tuple with the date of the article and the tokenized text.
    @in_dir: the directory of the articles
    '''
    alltext = {}
    date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|([A-Z][a-z]{2}\s\d{1,2},?\s\d{2,4})|(\d{4}[/-]\d{2}[/-]\d{2}|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b)"

    # Loop over all the articles
    for file_name in os.listdir(in_dir):
        if file_name.endswith('.txt'):
            input_path  = os.path.join(in_dir, file_name)
            with open(input_path, 'r') as infile:
                text = infile.read()

                # Find all matches of the date pattern in the text
                results = re.findall(date_pattern, text)

                # Parse the article date
                for i in results:
                    for j in i:
                        if j != '':
                            article_date = j
                            article_date = parser.parse(article_date)
                            article_date = article_date.strftime("%Y/%m/%d")

                # Results
                cleaned_text = clean_text(text)
                alltext[file_name] = [article_date, cleaned_text]

    return alltext

def vectorize_documents(article_dict):
    '''
    Vectorizes the dictionary of all the articles using TfidfVectorizer by sklearn.
    Outputs the transformed dataframe and the vectorizer.
    '''

    corpus = article_dict.values()
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(corpus)

    return X, vectorizer

def create_frequency_dataframe(only_output_dictionary=False):
    '''
    Creates a dataframe with all the words and their frequencies
    using CountVectorizer. 
    @only_output_dictionary boolean, if set to True, it will just output the dictionary with the words per article, without the date.
    else it will output the frequency dataframe.
    '''
    # Get all the cleaned text from the articles
    in_dir = "../data_prep/articles"
    alltext = all_text(in_dir)
    nodate = {}
    dates = {}

    # Create two seperate dictionaries, one with the date and one without
    for key, value in alltext.items():
        nodate[key] = value[1]
        dates[key] = value[0]
    
    if only_output_dictionary == False:
        # Use CountVectorizer to create a dataframe 
        # with all the words as columns and their frequencies as values
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(list(nodate.values()))
        df = pd.DataFrame(X.todense(), index=nodate.keys(), columns=vectorizer.get_feature_names_out())

        # Also add the data as column sorted by date
        df['Date'] = dates
        df = df.sort_values(by='Date')

        return df
    else:
        return nodate
        


def plot_timeline(dataframe, keywords):
    '''
    Plots a histogram of the occurances of user-inputted keywords over time.
    @dataframe dataframe as outputted by the function create_frequency_dataframe
    @keywords user defined keywords
    '''

    keywords.append('Date')
    # Filter dataframe on the keywords
    df = dataframe[keywords]

    # Transform the dataframe to have only a single 'Word' and 'Frequency' column
    df_long = df.melt(id_vars='Date', var_name='Word', value_name='Frequency')

    # Plot the data
    fig = px.histogram(df_long, x='Date', y='Frequency', title='Occurance of keywords', color='Word')
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()


def elbow_method(data):
    """
    This function uses the elbow method to determine the optimal number of clusters
    for the k-means clustering algorithm on a given dataset.
    
    @data pandas dataframe containing the data.
    """
    # Create a range of possible number of clusters
    n_clusters = range(1, 20)
    
    # Initialize an empty list to store the sum of squared distances for each k value
    sse = []
    
    # Loop through each k value and fit the k-means algorithm to the data
    for k in n_clusters:
        kmeans = KMeans(n_clusters=k, max_iter=600, random_state=0)
        kmeans.fit(data)
        
        # Compute the sum of squared distances for the fitted k-means model
        sse.append(kmeans.inertia_)
    
    # Plot the elbow curve to visualize the optimal number of clusters
    plt.plot(n_clusters, sse, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method')
    plt.show()
 

def plot_tsne_pca(show_clusters=True):
    """
    Plots two dimensionality reduction techniques, a TSNE-plot and a Principal Component Analysis plot,
    for a given corpus of text data. 
    
    @show_clusters : bool, Whether to show the plot with or without clustering. If True, the plot is colored by cluster labels.
    Returns fig : The PCA plot with or without cluster labels, depending on the value of `show_clusters`.
            fig2 : The TSNE plot with cluster labels.
    """

    # Get article text
    in_dir = "../data_prep/articles"
    alltext = all_text(in_dir)
    corpus = {}

    # Remove date
    for key, value in alltext.items():
        corpus[key] = value[1]
    
    corpus = corpus.values()

    # Initialize KMeans and Tfidf Vectorizer
    KM = KMeans(n_clusters=12, random_state=0, n_init="auto")
    vectorizer = TfidfVectorizer()

    # Transform the corpus, and predict each cluster
    X = vectorizer.fit_transform(corpus)
    KM.fit(X)
    prediction = KM.predict(X)

    pca = PCA(n_components=2).fit_transform(np.asarray(X.todense()))
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(np.asarray(X.todense())))

    # Get the most important keywords per cluster
    cluster_keywords = get_top_keywords(X, prediction, vectorizer.get_feature_names_out(), 5)
    new_labels = [cluster_keywords[i] for i in prediction]
   
    if show_clusters==False:
        fig = px.scatter(x=pca[:, 0], y=pca[:, 1], title="PCA Cluster plot")
    else:
        fig = px.scatter(x=pca[:, 0], y=pca[:, 1], color=new_labels, title="PCA Cluster plot")
    
    fig2 = px.scatter(x=tsne[:, 0], y=tsne[:, 1], color=new_labels, title="TSNE Cluster plot")
    
    return fig, fig2

    
def get_top_keywords(data, clusters, labels, n_terms):
    """
    This function returns the most important keywords per cluster of articles.
    @data: Tf-idf dataframe 
    @clusters: the clusters as outputted by the prediction of the KMeans algorithm
    @labels: used feature names of the vectorizer
    @n_terms: number of keywords to include
    """
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    keyword_dict = {}
    for i,r in df.iterrows():
        keyword_dict[i] = ', '.join([labels[t] for t in np.argsort(r)[-n_terms:]])
        
    return keyword_dict


def get_similar_articles(word, article_dict, X, vectorizer ):
    """
    This function computes the most similar artices for a user input based on the cosine similarity.

    @word: user input
    @article_dict: dictionary of all the articles and their text
    @X: Tf-idf dataframe
    @vectorizer: the vectorizer object
    """
    # Transform the user input using the Tf-idf vectorizer
    query_vector = vectorizer.transform([word])

    # Compute the cosine similarity between the query vector and all article vectors
    similarity_scores = cosine_similarity(X, query_vector)

    # Get the indices of the most similar articles
    most_similar_indices = similarity_scores.argsort(axis=0)[::-1][:10]

    article_titles = list(article_dict.keys())

    most_similar_articles = [(article_titles[int(i[0])], similarity_scores[i][0][0]) for i in most_similar_indices]

    # return the titles of the most similar articles
    return most_similar_articles

def classify_cluster(word, model, X, vectorizer, original_predictions):
    """
    This function classifies which cluster the user input is most likely to belong to.
    Outputs the predicted cluster number and the corresponding top keywords belonging to that cluster.

    @word: user input
    @model: KMeans model
    @X: Tf-idf dataframe
    @vectorizer: the vectorizer object
    @original_predictions: the clusters as outputted by the original prediction of the KMeans algorithm
    """
    # Transform the user input using the Tf-idf vectorizer
    query_vector = vectorizer.transform([word])

    cluster_keywords = get_top_keywords(X, original_predictions, vectorizer.get_feature_names_out(), 5)
    preprocess = np.asarray(query_vector.todense())
    prediction = model.predict(preprocess)
    
    return prediction[0], cluster_keywords
