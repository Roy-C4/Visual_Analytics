import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import networkx as nx
import dash_cytoscape as cyto
import math
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import dash_table
import os

"""
This file produces a star graph that helps us understand how each of the employees are conected to each other and 
what type of emails they exchanged. It also visualises the details of the senders so that the police can check if 
or not that person is suspicious based on their background and line of work. This helps them correlate with the other data visualised.
For this we have preprocessed the data previously to extract important information out of them. 

"""




#Read the data
df = pd.read_csv('../star_graph/new_email.csv')
df2 = pd.read_csv('../star_graph/EmployeeRecords.csv')


#Get the sender list
senders = df['From'].tolist()
sender_set = set(senders)



dictionary = {}


#Create a dictionary for sender 
for index, row in df.iterrows():
    key = row['From']
    value = row['Receivers']
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)

# Define the NetworkX directed graph for each sender
graphs = {}

for sender, receivers in dictionary.items():
#for sender in senders:
    G = nx.DiGraph()
    # Add the sender node
    G.add_node(sender)
    # Add the receiver nodes
    for receiver in receivers:
        G.add_node(receiver)
        # Add a directed edge from the sender to the receiver
        G.add_edge(sender, receiver)
    graphs[sender] = G

# print(graphs)
#Define the initial sender and graph to display
initial_sender = senders[0]
initial_graph = graphs[initial_sender]

# Convert the NetworkX graph to Cytoscape-compatible elements
elements = nx.cytoscape_data(initial_graph)['elements']

# Define the stylesheet for the graph
stylesheet = [
    {
    'selector': 'node',
    'style': {
        'background-color': 'blue',
        'label': 'data(id)',
        'font-size': '18px', # Increase the font size of the labels
        'text-halign': 'center',
        'text-valign': 'center',
        'text-margin-x': '5px',
        'text-margin-y': '-25px' # Move the label up
        }
    },
    {
    'selector': 'edge',
    'style': {
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'line-color': 'gray', # Change the line color to gray
        'target-arrow-color': 'gray' # Change the arrow color to gray
        }
    },
    
    {
        'selector': '.receiver',
        'style': {
            'background-color': 'green'
        }
    },
    {
    'selector': '.dangerous',
    'style': {
        'background-color': 'red',
        'font-size': '20px',
    }
}
]

# Define the dropdown for selecting a sender
sender_dropdown = html.Div([
    dbc.Label('Select a sender:'),
    dcc.Dropdown(
        id='sender-dropdown',
        options=[{'label': s, 'value': s} for s in sender_set],
        value=initial_sender
    )
], style={'margin-top': '20px'})

# Define the app layout
app_layout = dbc.Container([
    dbc.Row([
        dbc.Col(sender_dropdown, width=3),
        dbc.Col(
            cyto.Cytoscape(
                id='cytoscape-graph',
                elements=elements,
                layout={
                            'name': 'circle',
                            'radius': 250,
                            'startAngle': math.pi * 2,
                            'sweep': math.pi * 3 / 2,
                            'mouseoverEdgeData': {
                                'enabled': True
                                },
                            'mouseoverNodeData': {
                                'enabled': True
                                },
                            'responsive': True,
                            'minZoom': 0.5 
                        },
                
                style={'width': '100%', 'height': '400px'},
                stylesheet=stylesheet
            ),
            width=9
        )
    ]),
    
    dbc.Row([
        dbc.Col(
            html.Div(id='email-info'), 
            width=12,
            style={'margin-top': '20px'}
        ),
    ])
])


# Define the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = app_layout


@app.callback(
    [dash.dependencies.Output('cytoscape-graph', 'elements'),
      dash.dependencies.Output('email-info', 'children')],
    [dash.dependencies.Input('sender-dropdown', 'value')]
)


def update_graph(sender):
    if sender is None:
        raise PreventUpdate
        
    G = graphs[sender]
    elements = nx.cytoscape_data(G)['elements']
    
    # Get the dates, subjects, and receivers for the selected sender
    sender_data = df[df['From'] == sender]
    dates = sender_data['Date'].tolist()
    subjects = sender_data['Subject'].tolist()
    receivers = sender_data['Receivers'].tolist()
    
    # Combine the dates, subjects, and receivers into a table
    table_data = [{'Date': date, 'Subject': subject, 'Receivers': receiver} for date, subject, receiver in zip(dates, subjects, receivers)]
    
    # Get the employee record for the selected sender
    employee_data = df2[df2['EmailAddress'] == sender]
    
    # Combine the employee record and email table into a div
    employee_table = dash_table.DataTable(
        data=employee_data.to_dict('records'),
        columns=[{'name': k, 'id': k} for k in employee_data.columns],
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(230, 230, 230)'}
    )
    
    email_table = dash_table.DataTable(
        data=table_data,
        columns=[{'name': k, 'id': k} for k in table_data[0].keys()],
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(230, 230, 230)'}
    )
    
    
    div = html.Div([employee_table, email_table])
    
    return elements, div



if __name__ == '__main__':
    app.run_server(debug=True)






































# df = pd.read_csv('D:/roy/selected/TU/books/visual_analytics/New folder (2)/Disappearance at GAStech/data/graph/new_email.csv')

# senders = df['From'].tolist()
# sender_set = set(senders)


# dictionary = {}

# for index, row in df.iterrows():
#     key = row['From']
#     value = row['Receivers']
#     if key not in dictionary:
#         dictionary[key] = []
#     dictionary[key].append(value)
# # print(dictionary)


# # Define the senders and receivers
# # senders = ['sender1', 'sender2', 'sender3']
# # receivers = ['receiver1', 'receiver2', 'receiver3']

# # Define the NetworkX directed graph for each sender
# graphs = {}

# for sender, receivers in dictionary.items():
# #for sender in senders:
#     G = nx.DiGraph()
#     # Add the sender node
#     G.add_node(sender)
#     # Add the receiver nodes
#     for receiver in receivers:
#         G.add_node(receiver)
#         # Add a directed edge from the sender to the receiver
#         G.add_edge(sender, receiver)
#     graphs[sender] = G

# # print(graphs)
# #Define the initial sender and graph to display
# initial_sender = senders[0]
# initial_graph = graphs[initial_sender]

# # Convert the NetworkX graph to Cytoscape-compatible elements
# elements = nx.cytoscape_data(initial_graph)['elements']

# # Define the stylesheet for the graph
# stylesheet = [
#     {
#     'selector': 'node',
#     'style': {
#         'background-color': 'blue',
#         'label': 'data(id)',
#         'font-size': '18px', # Increase the font size of the labels
#         'text-halign': 'center',
#         'text-valign': 'center',
#         'text-margin-x': '5px',
#         'text-margin-y': '-25px' # Move the label up
#         }
#     },
#     {
#     'selector': 'edge',
#     'style': {
#         'curve-style': 'bezier',
#         'target-arrow-shape': 'triangle',
#         'line-color': 'gray', # Change the line color to gray
#         'target-arrow-color': 'gray' # Change the arrow color to gray
#         }
#     },
#     {
#         'selector': 'node[id="' + initial_sender + '"]',
#         'style': {
#             'background-color': 'red',
#             'font-size': '20px',
#         }
#     },
#     {
#         'selector': '.receiver',
#         'style': {
#             'background-color': 'green'
#         }
#     }
# ]

# # Define the dropdown for selecting a sender
# sender_dropdown = html.Div([
#     dbc.Label('Select a sender:'),
#     dcc.Dropdown(
#         id='sender-dropdown',
#         options=[{'label': s, 'value': s} for s in sender_set],
#         value=initial_sender
#     )
# ], style={'margin-top': '20px'})

# # Define the app layout
# app_layout = dbc.Container([
#     dbc.Row([
#         dbc.Col(sender_dropdown, width=3),
#         dbc.Col(
#             cyto.Cytoscape(
#                 id='cytoscape-graph',
#                 elements=elements,
#                 layout={
#                             'name': 'circle',
#                             'radius': 250,
#                             'startAngle': math.pi * 2,
#                             'sweep': math.pi * 3 / 2,
#                             'mouseoverEdgeData': {
#                                 'enabled': True
#                                 },
#                             'mouseoverNodeData': {
#                                 'enabled': True
#                                 },
#                             'responsive': True,
#                             'minZoom': 0.5 
#                         },
                
#                 style={'width': '100%', 'height': '400px'},
#                 stylesheet=stylesheet
#             ),
#             width=9
#         )
#     ]),
    
#     dbc.Row([
#         dbc.Col(
#             html.Div(id='email-info'), 
#             width=12,
#             style={'margin-top': '20px'}
#         ),
#     ])
# ])


# # Define the app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app.layout = app_layout

# @app.callback(
#     [dash.dependencies.Output('cytoscape-graph', 'elements'),
#      dash.dependencies.Output('email-info', 'children')],
#     [dash.dependencies.Input('sender-dropdown', 'value')]
# )
# def update_graph(sender):
#     if sender is None:
#         raise PreventUpdate
        
#     G = graphs[sender]
#     elements = nx.cytoscape_data(G)['elements']
    
#     # Get the dates and subjects for the selected sender
#     sender_data = df[df['From'] == sender]
#     dates = sender_data['Date'].tolist()
#     subjects = sender_data['Subject'].tolist()
    
#     # Combine the dates and subjects into a list of html.Div elements
#     email_info = [html.Div(f'{date}: {subject}') for date, subject in zip(dates, subjects)]
    
#     return elements, email_info

# if __name__ == '__main__':
#     app.run_server(debug=True)




