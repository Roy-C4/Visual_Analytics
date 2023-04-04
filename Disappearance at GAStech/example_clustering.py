import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define the directory containing the text files to be clustered
text_dir = 'D:\roy\selected\TU\books\visual_analytics\Disappearance at GAStech\data\data_prep\articles'

# Load the text data into a list
docs = []
for filename in os.listdir(text_dir):
    with open(os.path.join(text_dir, filename), 'r') as f:
        docs.append(f.read())

# Tokenize and vectorize the text data using TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents=string.punctuation)
X = tfidf.fit_transform(docs)

# Cluster the text data using k-means clustering
k = 3 # Set the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)

# Reduce the dimensionality of the feature vectors using t-SNE
X_tsne = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(X)

# Visualize the clusters using a scatter plot
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_)
plt.title('Text File Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
