# If you prefer to run the code online instead of on your computer click:
# https://github.com/Coding-with-Adam/Dash-by-Plotly#execute-code-in-browser

from dash import Dash, dcc, Output, Input, html  # pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import data_prep
import plotly.express as px


# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])
# mytitle = dcc.Markdown(children='# Mail activity per employee over time')
# mygraph = dcc.Graph(figure={})

df = data_prep.create_frequency_dataframe()
options = [{'label': 'death', 'value':'death'},
    	{'label': 'fire',  'value':'fire'},
        {'label': 'pok',  'value':'pok'}, 
        {'label': 'police', 'value':'police'}]
keywords = ['death', 'fire', 'pok', 'police']
mytitle = dcc.Markdown(children='# Mail activity per employee over time')
mygraph = dcc.Graph(figure={})
dropdown = dcc.Dropdown(options=['Scatter Plot'],
                        value='Scatter Plot',  # initial value displayed when page first loads
                        clearable=False)

# Customize your own Layout
app.layout = html.Div(
    [
        dcc.Checklist(
            options=options,
            inline=True,
            value=keywords,
            id="checklist")
        ,
        dcc.Graph(id="stackedbar")
    ]
    
)
 

# Callback allows components to interact
@app.callback(
    Output("stackedbar", component_property='figure'),
    Input("checklist", component_property='value')
)
def update_graph(values):  # function arguments come from the component property of the Input
    
    keywords.append('Date')

    df_long = df.melt(id_vars='Date', var_name='Word', value_name='Frequency')
    df_long = df_long[df_long['Word'].isin(values)]

    fig = px.histogram(df_long, x='Date', y='Frequency', title='Occurance of keywords', color='Word')
    fig.update_xaxes(rangeslider_visible=True)
    
    
    return fig  # returned objects are assigned to the component property of the Output


# Run app
if __name__=='__main__':
   app.run_server(port=8053)