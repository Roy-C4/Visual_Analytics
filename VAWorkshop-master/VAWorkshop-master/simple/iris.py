from typing import Dict

import pandas as pd
from dash import dash
from dash import Dash, dcc, html, Input, Output, State, ctx
from sklearn import metrics
import plotly.express as px
from sklearn.cluster import KMeans
import dash_bootstrap_components as dbc

#%%
df = px.data.iris()

#%%

def simple_k_means(X: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_transform(X)

    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(X, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


no_species_column = simple_k_means(df.iloc[:, :4])
l = no_species_column['model'].labels_

df2 = df[:]
df2['label'] = l
dropdown_values = [dict(label=x, value=i) for i, x in enumerate(list(df.columns[:-1]))]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1(children='Hello Dash'),
    html.Div(children='Dash: A web application framework for Python.'),
    html.Hr(),
    html.Div([
        dbc.Label("Choose x"),
        dcc.Dropdown(id="dropdown_x", value=1, options=dropdown_values),
    ]),
    html.Div([
        dbc.Label("Choose y"),
        dcc.Dropdown(id="dropdown_y", value=1, options=dropdown_values),
    ]),
    dbc.Button('Run Callback', id='example-button', color='primary', style={'margin-bottom': '1em'}),
    dbc.Row([
        dbc.Col(dcc.Graph(id='example-graph')),
    ])
])


@app.callback(
    Output('example-graph', 'figure'),
    [Input('example-button', 'n_clicks')],
    [State('dropdown_x', 'value'),
     State('dropdown_y', 'value')])
def update_figure(n_clicks, dropdown_x, dropdown_y):
    x = dropdown_values[dropdown_x]['label']
    y = dropdown_values[dropdown_y]['label']
    return px.scatter(df2, x=x, y=y, color='label')


# @app.callback(Output('slider-value', 'children'), [Input('slider', 'value')])
# def update_slider_value(slider):
#     return f'Multiplier: {slider}'


app.run_server(debug=True, port=3000, host='0.0.0.0')
