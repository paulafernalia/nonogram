import numpy as np
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import json


import sample_puzzle as sp
from utils import GridMap, plotly_heatmap

# Colour definitions
b = 1
w = 0
g = 0.2


game = GridMap(
    sp.col_rules, 
    sp.row_rules
    )

game._grid[0,0] = 1
game._grid[0,1] = 0


# Main layout
app = dash.Dash()
app.layout = html.Div([

    dcc.Dropdown(
        id='puzzle-selector',
        options=[
            {'label': 'Puzzle1', 'value': 'P1'},
            {'label': 'Puzzle2', 'value': 'P2'},
            {'label': 'Puzzle3', 'value': 'P3'}
        ],
        value = 'P2'
    ),

    dcc.Graph(
        id='heatmap',
        figure=go.Figure(

            data = plotly_heatmap(
                game._grid, game._w, game._h, 
                game._row_labels, game._col_labels)['data'],
            layout=plotly_heatmap(
                game._grid, game._w, game._h, 
                game._row_labels, game._col_labels)['layout'],
        ),
    ),

    html.Div(id='hidden-data', style={'display': 'none'}),

])



@app.callback(
    Output('hidden-data', 'children'),
    [Input('puzzle-selector', 'value'),
     Input('heatmap', 'clickData')],
    [State('hidden-data', 'children')])
def update_data(dropdown_value, clickData, game_json):

    update_all = 0

    if game_json is None:
        update_all = 1

    if game_json is not None:
        game_dict = json.loads(game_json)
        if dropdown_value != game_dict['puzzle']:
            update_all = 1

    if update_all == 1:

        game = GridMap(
            sp.col_rules, 
            sp.row_rules
        )

        game._grid[0,0] = 1
        game._grid[0,1] = 0

        game_dict = {}
        game_dict['col_rules'] = game._col_rules
        game_dict['row_rules'] = game._row_rules
        game_dict['row_labels'] = game._row_labels
        game_dict['col_labels'] = game._col_labels
        game_dict['w'] = game._w
        game_dict['h'] = game._h
        game_dict['grid'] = game._grid.tolist()
        game_dict['puzzle'] = dropdown_value

        game_dict['changes'] = {
            "0.5": 1, # White to black
            "1": 0,   # Black to grey 
            "0": 0.5 # Grey to white
        }
        

    elif clickData is not None:
        grid = np.array(game_dict['grid'])
        
        x = clickData["points"][0]["x"]
        y = clickData["points"][0]["y"]
        z = clickData["points"][0]["z"] 

        grid[y, x] = game_dict['changes'][str(z)]
        game_dict['grid'] = grid.tolist()

    
    return json.dumps(game_dict)

        


@app.callback(
    Output('heatmap', 'figure'),
    [Input('hidden-data', 'children')])
def update_heatmap(game_json):

    game_dict = json.loads(game_json)
    grid = np.array(game_dict['grid'])

    heatmap= plotly_heatmap(
        grid, game_dict['w'], game_dict['h'], 
        game_dict['row_labels'], game_dict['col_labels'])

    figure = go.Figure(
        data =heatmap['data'],
        layout=heatmap['layout']
    )

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)