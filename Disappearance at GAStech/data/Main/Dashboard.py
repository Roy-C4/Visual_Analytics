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




app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Global dataframe for the word frequencies
df = prepare_plots.create_frequency_dataframe()

list_value = ['No similar articles']

options = [{'label': 'Death', 'value':'death'},
    	{'label': 'Fire',  'value':'fire'},
        {'label': 'Pok',  'value':'pok'}]
keywords = ['death', 'fire', 'pok']


## Help
app.layout = html.Div([ 
    html.Div([
        dcc.Checklist(
            options=options,
            inline=True,
            value=keywords,
            id="checklist", 
            inputStyle={"margin-left": "10px", "margin-right": "5px"})
        ,
        dcc.Graph(
            id="stackedbar", 
            figure={}) 
        ], style={'margin':100}),
    
    html.Div([
        html.H1(
            id='User input', 
            style={'textAlign': 'center', 'margin':25}
            ),
        html.Br()
        , 
        dcc.Input(
            id='input-box', 
            type='text', 
            value='Search for a word!', 
            debounce=True
            )],
            style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
        ),
    html.Div([
        dcc.Dropdown(
            ['PCA', 'tSNE'], 
            value='PCA', 
            id='dropdown-clusters'
            ),
        ], style={'padding': '50px'}),
    html.Div([
        html.Div(
        className="trend",
        children=[
            html.Ul(id='my-list', children=[html.Li(i) for i in list_value])
            ],
        ),
        dcc.Graph(
            id="article_clustering", 
            figure={}, 
            style={'width':'50%', 'margin':25, 'display': 'inline-block'}
            )
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
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
    pca, tsne = prepare_plots.plot_tsne_pca()
    if input == 'tSNE':
        return tsne
    else:
        return pca
    
@app.callback(
    Output("my-list", "children"),
    Input("input-box", "value"))

def plot_similar_articles(word):
    if word:
        similar_articles = prepare_plots.get_similar_articles(word)
        return [html.Li(i[0], str(i[1])) for i in similar_articles]
    




# Run app
if __name__=='__main__':
   app.run_server(port=8053, debug=True)