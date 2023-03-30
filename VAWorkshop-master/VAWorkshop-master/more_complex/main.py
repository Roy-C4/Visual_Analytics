from typing import Dict, List

import pandas as pd
from dash import dash
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

import plotly.figure_factory as ff
import plotly.express as px

from more_complex.a_preprocessing import read
from more_complex.b_ml import do_ml
from more_complex.c_dr import run_dr

df = read()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1(children='Hello Dash'),
    html.Div(children='Dash: A web application framework for Python.'),
    html.Hr(),
    dbc.Button('Run Callback', id='example-button', color='primary', style={'margin-bottom': '1em'}),
    dbc.Row([
        dbc.Col([
            dbc.Checklist(
                [{"label": x, "value": x} for x in df.columns],
                value=[x for x in df.columns], id="checklist-input",
            )
        ], width=3),
        dbc.Col(dcc.Graph(id='example-graph', style={'height': '50vh'})),
    ])
], fluid=True)


@app.callback(
    Output('example-graph', 'figure'),
    [Input('example-button', 'n_clicks')],
    [State('checklist-input', 'value')])
def update_figure(n_clicks, checklist):
    print(checklist)
    checklist_remove = [x for x in df.columns if x not in checklist]
    remove = ['Country name', 'Standard error of ladder score']
    target = ['Ladder score']
    gnb, X_train, X_test, y_train, y_test, X, y = do_ml(df, remove+checklist_remove, target)
    dr, projected = run_dr(df, remove + target + checklist_remove)
    projected = pd.DataFrame(projected)
    projected['predict'] = gnb.predict(X)
    projected['actual'] = y
    projected['diff'] = (projected['predict'] - projected['actual']).abs()
    return px.scatter(pd.concat([projected, df], axis=1), x=0, y=1, color='diff', hover_data=df.columns, size='diff')
    # return px.scatter(pd.concat([projected, df], axis=1), x=0, y=1, color='Regional indicator', hover_data=df.columns, size='diff')
    # return None


app.run_server(debug=True, port=3000, host='0.0.0.0')
