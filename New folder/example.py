# import dash
# import dash_bootstrap_components as dbc
# import dash_html_components as html
# import dash_core_components as dcc

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# app.layout = html.Div(
#     [
#         dbc.Row(
#             dbc.Col(html.H1("My Dashboard"), width={"size": 6, "offset": 3}),
#         ),
#         dbc.Row(
#             [
#                 dbc.Col(
#                     dcc.Graph(id="graph-1", figure=my_fig),
#                     width={"size": 6, "order": "first"},
#                 ),
#                 dbc.Col(
#                     dcc.Graph(id="graph-2", figure=my_fig),
#                     width={"size": 6, "order": "last"},
#                 ),
#             ],
#             align="center",
#         ),
#     ]
# )



# if __name__ == "__main__":
#     app.run_server(debug=True)



# import dash
# import dash_bootstrap_components as dbc
# import dash_core_components as dcc
# import dash_html_components as html
# import plotly.express as px
# import pandas as pd

# # Load data
# df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv")

# # Create app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# # Define layout
# app.layout = dbc.Container(
#     [
#         dbc.Row(
#             dbc.Col(
#                 html.H1("Apple Stock Prices"),
#                 width={"size": 6, "offset": 3},
#             ),
#         ),
#         dbc.Row(
#             dbc.Col(
#                 dcc.Graph(
#                     id="apple-stock-graph",
#                     figure=px.bar(df, x="AAPL_x", y="AAPL_y"),
#                 ),
#                 width={"size": 8, "offset": 2},
#             ),
#         ),
#     ],
#     fluid=True,
# )

# # Run app
# if __name__ == "__main__":
#     app.run_server(debug=True)




import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px


# # Load data
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

# # Initialize the app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# # Define the sidebar navigation
# sidebar = html.Div(
#     [
#         dbc.Nav(
#             [
#                 dbc.NavLink("Page 1", href="/page-1", id="page-1-link"),
#                 dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
#             ],
#             vertical=True,
#             pills=True,
#         ),
#     ],
#     className="sidebar",
# )

# # Define the content of the app
# content = html.Div(id="page-content", className="content")

# # Define the layout of the app
# app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# # Define the callback for updating the page content
# @app.callback(
#     [dash.dependencies.Output("page-content", "children")],
#     [dash.dependencies.Input("url", "pathname")],
# )
# def render_page_content(pathname):
#     if pathname == "/page-1":
#         # Create the bar chart
#         bar_chart = dcc.Graph(
#             id='bar-chart',
#             figure={
#                 'data': [
#                     go.Bar(
#                         x=df['State'],
#                         y=df['Population'],
#                     )
#                 ],
#                 'layout': go.Layout(
#                     title='Population by State',
#                     xaxis={'title': 'State'},
#                     yaxis={'title': 'Population'}
#                 )
#             }
#         )
        
#         # Create the scatter plot
#         scatter_plot = dcc.Graph(
#             id='scatter-plot',
#             figure={
#                 'data': [
#                     go.Scatter(
#                         x=df['Murder.Rate'],
#                         y=df['Population'],
#                         mode='markers'
#                     )
#                 ],
#                 'layout': go.Layout(
#                     title='Murder Rate vs. Population',
#                     xaxis={'title': 'Murder Rate'},
#                     yaxis={'title': 'Population'}
#                 )
#             }
#         )
        
#         # Return the page content
#         return [
#             html.H1('Page 1'),
#             html.H2('Bar Chart'),
#             bar_chart,
#             html.H2('Scatter Plot'),
#             scatter_plot
#         ]
    
#     elif pathname == "/page-2":
#         # Create the line chart
#         line_chart = dcc.Graph(
#             id='line-chart',
#             figure={
#                 'data': [
#                     go.Scatter(
#                         x=df['State'],
#                         y=df['Murder.Rate'],
#                         mode='lines'
#                     )
#                 ],
#                 'layout': go.Layout(
#                     title='Murder Rate by State',
#                     xaxis={'title': 'State'},
#                     yaxis={'title': 'Murder Rate'}
#                 )
#             }
#         )
        
#         # Return the page content
#         return [
#             html.H1('Page 2'),
#             html.H2('Line Chart'),
#             line_chart
#         ]
    
#     else:
#         return []

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)




# load data
df = pd.read_csv('D:/roy/selected/TU/books/visual_analytics/Disappearance at GAStech/data/word_cloud/iranian_students.csv')

# initialize the app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# create a function that returns the layout for page 1
def layout_page_1():
    # create the bar chart
    fig_bar = px.bar(df, x='Years', y='Girls Grade School', #color='State', 
                     labels={'Years': 'Years', 'Girls Grade School': 'Girls Grade School'}, 
                     title='Girls in grade school per YEar in Iran')
    # create the scatter plot
    fig_scatter = px.scatter(df, x='Years', y='Boys Grade School', #color='State',
                             labels={'Years': 'Years', 'Boys Grade School': 'Boys Grade School'},
                             title='Boys in grade school per YEar in Iran')
    # create the layout for page 1
    layout = html.Div([
        dbc.Container([
            html.H1('Page 1'),
            html.Hr(),
            html.H2('Murder per State'),
            dcc.Graph(id='graph-bar', figure=fig_bar),
            html.H2('Murders vs Rapes per State'),
            dcc.Graph(id='graph-scatter', figure=fig_scatter)
        ])
    ])
    return layout

# create a function that returns the layout for page 2
def layout_page_2():
    # create the line chart
    fig_line = px.line(df, x='Years', y='', #color='State',
                       labels={'Years': 'Years', 'Girls Middle School': 'Girls Middle School'},
                       title='Population per State in Connecticut and Mississippi')
    # create the layout for page 2
    layout = html.Div([
        dbc.Container([
            html.H1('Page 2'),
            html.Hr(),
            html.H2('Population per State'),
            dcc.Graph(id='graph-line', figure=fig_line)
        ])
    ])
    return layout

# create the app layout
app.layout = html.Div([
    # create the sidebar navigation
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Page 1", href="/page-1")),
            dbc.NavItem(dbc.NavLink("Page 2", href="/page-2")),
        ],
        brand="My Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    # create the content area
    html.Div(id='page-content')
])

# create the callback function that changes the page layout based on the URL
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return layout_page_1()
    elif pathname == '/page-2':
        return layout_page_2()
    else:
        return layout_page_1()

# run the app
if __name__ == '__main__':
    app.run_server(debug=True, suppress_callback_exceptions=True)



   

