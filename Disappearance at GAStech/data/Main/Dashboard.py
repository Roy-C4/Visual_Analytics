# If you prefer to run the code online instead of on your computer click:
# https://github.com/Coding-with-Adam/Dash-by-Plotly#execute-code-in-browser

from dash import Dash, dcc, Output, Input, html  # pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import prepare_plots
import plotly.express as px




app = Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])


# Global dataframe for the word frequencies
df = prepare_plots.create_frequency_dataframe()
options = [{'label': 'death', 'value':'death'},
    	{'label': 'fire',  'value':'fire'},
        {'label': 'pok',  'value':'pok'}]
keywords = ['death', 'fire', 'pok']


app.layout = html.Div([
    html.H1('User input'),
    dcc.Input(id='input-box', type='text', value='', debounce=True),
    dcc.Checklist(
        options=options,
        inline=True,
        value=keywords,
        id="checklist")
    ,
    dcc.Graph(id="stackedbar", figure={})
    ]
    
)
 
# Callback for updating Word-Frequency graph based on checklist
@app.callback(
    Output("stackedbar", component_property='figure'),
    Input("checklist", component_property='value')
)

def update_graph(values):  
    # Reshape dataframe to get a column with the word names as values and a column with the frequency
    df_long = df.melt(id_vars='Date', var_name='Word', value_name='Frequency')
    
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
            options.append({'label': value, 'value': value})
            return options    

    else:
        return options   
        
    



# Run app
if __name__=='__main__':
   app.run_server(port=8053, debug=True)