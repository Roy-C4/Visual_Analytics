# If you prefer to run the code online instead of on your computer click:
# https://github.com/Coding-with-Adam/Dash-by-Plotly#execute-code-in-browser

from dash import Dash, dcc, Output, Input, html  # pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import prepare_plots
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd



app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


colors = {
    'background_element': '#516096',
    'text': '#fafafc'
    'background'
}
# Global dataframe for the word frequencies
df = prepare_plots.create_frequency_dataframe()
pca, tsne = prepare_plots.plot_tsne_pca()
model = KMeans(n_clusters=12, random_state=0, n_init="auto")
article_dict = prepare_plots.create_frequency_dataframe(only_output_dictionary=True)
X, vectorizer = prepare_plots.vectorize_documents(article_dict)
model.fit(X)
original_predictions = model.predict(X)

list_value = ['No similar articles']

options = [{'label': 'Death', 'value':'death'},
    	{'label': 'Fire',  'value':'fire'},
        {'label': 'Pok',  'value':'pok'}]
keywords = ['death', 'fire', 'pok']


## Help
app.layout = html.Div([ 
    html.Div([
    html.H1('Article Analysis & Clustering', style={'padding':'20px', 'textAlign': 'center'})
    ]),

    dbc.Row([
        html.Div([
            dcc.Checklist(
                options=options,
                inline=True,
                value=keywords,
                id="checklist", 
                inputStyle={"margin-left": "10px", "margin-right": "5px"}
            ),
            dcc.Graph(
                id="stackedbar", 
                figure={}
            ) 
        ], style={'margin':'auto'}),
    ]),

    dbc.Row([
    # Add the user input box centered in the middle
        html.Div([
            html.H2('Enter your keyword:', style={'padding':'20px', 'border': '1px solid smokegray'}),
            dcc.Input(id='input-box', 
                    type='text', 
                    value=''
            ),
        ], style={'textAlign': 'center'}),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Classified cluster: ", id='cluster_classification', style={'border': '1px solid smokegray', 'text-align': 'center'}),
                html.H4("Cluster top keywords: ", id='cluster_classification_keywords', style={'border': '1px solid smokegray', 'text-align': 'center'})
            ]),
            dcc.Graph(
                id="similar_articles", 
                figure={}, 
                style={'width':'100%', 'margin':25, 'border': '1px solid smokegray'}
            )
        ], width=4),
        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    options=[{'label': 'PCA', 'value': 'PCA'},
                            {'label': 'tSNE', 'value': 'tSNE'}], 
                    value='PCA', 
                    id='dropdown-clusters'
                )
            ], style={'padding': '20px'}),
            dcc.Graph(
                id="article_clustering", 
                figure={}, 
                style={'width':'100%', 'margin':25, 'border': '1px solid smokegray'}
            )
        ], width=8)
    ], align='center', justify='center')


 
])

# Callback for updating Word-Frequency graph based on checklist
@app.callback(
    Output("stackedbar", component_property='figure'),
    Input("checklist", component_property='value')
)

def update_graph(values):  
    # Reshape dataframe to get a column with the word names as values and a column with the frequency
    df_index_reset = df.reset_index()
    
    df_long = df_index_reset.melt(id_vars=['Date', 'level_0'], var_name='Word', value_name='Frequency')
    df_long.rename(columns={'level_0':'Article_ID'}, inplace=True)
    # Get only the rows of the words that we want
    df_long = df_long[df_long['Word'].isin(values)]
    
    # Plot the graph with rangeslider
    fig = px.histogram(df_long, x='Date', y='Frequency', title='Occurance of keywords', color='Word')
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig  

# Callback for updating the checklist based on user input
@app.callback(
    Output('checklist', component_property='options'),
    [Input('input-box', component_property='value')]
)

def update_checklist(input_value):
    if input_value:
        value = str(input_value).lower()
        # Remove any whitespace 
        value = value.strip()

        # Get the vocabulary
        all_words = df.columns.tolist()

        # If user input is in the vocabulary, update the checklist options
        if value not in all_words:
            return options
        else:
            options.append({'label': value.capitalize(), 'value': value})
            return options    

    else:
        return options   
    

@app.callback(
    Output("article_clustering", "figure"),
    Input("dropdown-clusters", "value"),
)

def update_cluster_graphs(input): 
    if input == 'tSNE':
        return tsne
    else:
        return pca
    
@app.callback(
    Output("cluster_classification", "children"),
    Output("cluster_classification_keywords", "children"),
    Input("input-box", "value"))

def get_cluster_classification(word, model=model, X=X, vectorizer=vectorizer, original_predictions=original_predictions):
    if word != '':
        prediction, keywords = prepare_plots.classify_cluster(word, model, X, vectorizer, original_predictions)
        return f'Classified cluster: {prediction}', keywords[prediction]
    else:
        return 'No word chosen yet!', '-'
    
    
@app.callback(
    Output("similar_articles", "figure"),
    Input("input-box", "value"))

def get_similar_articles_graph(word, article_dict=article_dict, X=X, vectorizer=vectorizer):
    
    most_similar_articles = prepare_plots.get_similar_articles(word, article_dict, X, vectorizer)
    df = pd.DataFrame(most_similar_articles, columns=['Article Title', 'Cosine Similarity Score (%)'])
    df['Cosine Similarity Score (%)'] = df['Cosine Similarity Score (%)'].apply(lambda x: x*100)
    fig = px.bar(df, x='Cosine Similarity Score (%)', y='Article Title', title='Cosine Similariy for the chosen word', color='Article Title', color_discrete_sequence=px.colors.qualitative.Safe, orientation='h')

    return fig
        




# Run app
if __name__=='__main__':
   app.run_server(port=8053, debug=True)