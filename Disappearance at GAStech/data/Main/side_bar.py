# Code source: https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

import word_cloud
import plots

df = pd.read_csv('iranian_students.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])



sidebar = dbc.Card(
    [
        dbc.CardBody([    
        html.H2("Visualisations", className="display-8"),
        html.Hr(),
        html.P(
            "Number of students per education level", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
                dbc.NavLink("Page 3", href="/page-3", active="exact"),
                dbc.NavLink("Page 4", href="/page-4", active="exact"),
                
            ],
            vertical=True,
            pills=True,
        ),

    ])
    ],
    color = "light", style= {"height" : "100vh","width" : "20rem", "position" : "fixed" }
)

content = html.Div(
    html.H1('Kindergarten in Iran',
            style={'textAlign':'center'}, id='header'),
    dcc.Graph(id='bargraph',
              y=['Girls Kindergarten', 'Boys Kindergarten']),
    id="page-content", style={"padding" : "5rem"})


app.layout = dbc.Container([
    dcc.Location(id="url"),
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col(content, width=9, style={"margin-left" : "16rem"}) 
    ])
], fluid=True)


@app.callback(
    [Output("page-content", "children")],
    [Output('bargraph', ' figure'), Output('header', 'children')]
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/page-1":
        return px.bar(df, barmode='group', x='Years', y=['Girls Kindergarten', 'Boys Kindergarten'])
    elif pathname == "/page-2":
        return [
                html.H1('Grad School in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                          figure=px.bar(df, barmode='group', x='Years',
                          y=['Girls Grade School', 'Boys Grade School']))
                ]
    elif pathname == "/page-3":
        graph_type = plots.dropdown
        figg = 'Scatter Plot'
        if graph_type == 'Scatter Plot':
            figg = plots.fig
        #else:
            #fig = px.bar(df, barmode='group', x='Years', y=['Girls Kindergarten', 'Boys Kindergarten'])
            
        return[
        html.H1('Mail activity per employee over time', style={'textAlign': 'center'}),
        dcc.Graph(id='Email_graph', figure=figg)
    ]
        
    elif pathname == "/page-4":
        figg = word_cloud.fig
        return [
            html.H1('Word cloud for potential keywords', style={'textAlign':'center'}),
            dcc.Graph(id='wordcloud', figure=figg)
        ]
    # If the user tries to reach a different page, return a 404 message
    """return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )"""


if __name__=='__main__':
    app.run_server(debug=True, port=3000)



    

