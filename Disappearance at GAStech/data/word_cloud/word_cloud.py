import os
import string
import dash
import plotly.express as px
from dash import dcc
from dash import html
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from dash.dependencies import Input, Output

from flask import Flask

# Initialize Flask server
server = Flask(__name__)

# Set input directories
in_dir = r'../data_prep/output_article'
in_dir2 = r'../Background/names_pok/5 year report clean.txt'

 # create an empty list to store article names
options = [] 


# Iterate over files in the input directory
for file_name in os.listdir(in_dir):
    if file_name.endswith('.txt'):  # Check if file is a text file
        options.append(file_name)  # append the file name to the options list


# Sort the list of article names in ascending order
options = sorted(options, key=lambda x: int(x.split('.')[0]))

# Define a function to clean text
def clean_text(text):
    # Remove punctuation from text
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text into words and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Create a set of stop words
    stop_words = set(stopwords.words('english'))
    # Remove stop words and non-alphabetic words from the list of tokens
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Join the remaining words into a string and return
    return ' '.join(words)


# Open the PoK report file and generate a word cloud
with open(in_dir2, 'r') as infile:
    text = infile.read()
    cleaned_text = clean_text(text)
    word_cld_pok = WordCloud(width=800, height=800, background_color='white').generate(cleaned_text)
    fig_pok = px.imshow(word_cld_pok)
    fig_pok.update_layout()



# Initialize the Dash application
app = dash.Dash(__name__, server=server, url_base_pathname='/app1/')

# Define the layout of the Dash application
app.layout = html.Div([
    dcc.Dropdown(
        id='article-dropdown',
        options=[{'label': name, 'value': name} for name in options],  # create dropdown options
        value=options[0]  # set default value to first article
    ),
    dcc.Graph(id='word-cloud'),
    html.H2('Important people of PoK'),
    dcc.Graph(id='word-cloud-pok', figure=fig_pok)
])

# Define a callback function to update the word cloud when a new article is selected
@app.callback(
    Output('word-cloud', 'figure'),
    Output('word-cloud-pok', 'figure'),
    [Input('article-dropdown', 'value')]
)

#Generate and update wordcloud
def update_word_cloud(selected_article):
    if selected_article == '5 year.txt':
        with open(in_dir2, 'r') as infile:
            text = infile.read()
            cleaned_text = clean_text(text)
            word_cld_pok = WordCloud(width=800, height=800, background_color='white').generate(cleaned_text)
            fig_pok = px.imshow(word_cld_pok)
            fig_pok.update_layout()
            return None, fig_pok
    else:
        input_path = os.path.join(in_dir, selected_article)
        with open(input_path, 'r') as infile:
            text = infile.read()
            cleaned_text = clean_text(text)
            word_cld = WordCloud(width=800, height=800, background_color='white').generate(cleaned_text)
            fig = px.imshow(word_cld)
            fig.update_layout()
            return fig, None

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)


