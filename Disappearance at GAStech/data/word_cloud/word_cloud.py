import os
import string
import dash
import plotly.express as px

from dash import dcc
from dash import html
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

in_dir = r'D:\roy\selected\TU\books\visual analytics\1. Disappearance at GAStech\1. Disappearance at GAStech\data\output_article'


def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha()
             and word not in stop_words]
    return ' '.join(words)


def cloud(text):
    word_cld = WordCloud(width=800, height=800,
                         background_color='white').generate(text)
    fig = px.imshow(word_cld)
    fig.update_layout(title_text='Word Cloud')
    return fig


with open('0.txt', 'r') as infile:
    text = infile.read()


#cleaned_text = clean_text(text)

#fig = cloud(cleaned_text)

for file_name in os.listdir(in_dir):
    if file_name.endswith('.txt'):
        input_path  = os.path.join(in_dir, file_name)
        with open(input_path, 'r') as infile:
            text = infile.read()
            cleaned_text = clean_text(text)
            fig = cloud(cleaned_text)


app = dash.Dash()

app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
