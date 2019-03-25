import numpy as np
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import json
import pyomo.environ as pyo

import opt
import sample_puzzle as sp
from utils import GridMap, plotly_heatmap
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

# Main layout



def serve_layout():
    
    game = GridMap(
        sp.nono_dict['Nonogram 1: 10x10']['col_rules'], 
        sp.nono_dict['Nonogram 1: 10x10']['row_rules']
    )

    game_dict = {}
    game_dict['col_rules'] = game._col_rules
    game_dict['row_rules'] = game._row_rules
    game_dict['row_labels'] = game._row_labels
    game_dict['col_labels'] = game._col_labels
    game_dict['w'] = game._w
    game_dict['h'] = game._h
    game_dict['grid'] = game._grid.tolist()
    game_dict['puzzle'] = 'Nonogram 1: 10x10'
    game_dict['n_clicks'] = 0
    game_dict['n_clicks_clear'] = 0

    game_dict['changes'] = {
        "0.5": 0, # White to black
        "1": 0.5,   # Black to grey 
        "0": 1 # Grey to white
    }

    return html.Div([

        html.Div([
            html.Div([
                html.H3('Nonogram Solver', className='app-title'),

            ], className="three columns"),

            html.Div([
                dcc.Dropdown(
                    id='puzzle-selector',
                    options= [
                        {'label': nono, 'value': nono} for nono in sp.nono_dict.keys()
                    ],
                    value = 'Nonogram 1: 10x10',
                    style={
                        'marginTop': '15px', 
                        'color': '#0b1b47', 
                        'background': 'white',
                        'width': '300px',
                        'fontSize': '15px',
                        'fontFamily': 'sans-serif'}
                )
            ], 
            className="three columns"
            ),

            html.Div([
                html.Button('Solve', id='button', className='button')
            ], className="three columns"),

            html.Div([
                html.Button('Clear', id='button-clear', className='button')
            ], className="three columns"),
        ], 
        className="row",
        style={"background-color": "#9cbff4", "height": "70px"}
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

        html.Div(
            id='hidden-data', 
            children = json.dumps(game_dict),
            style={'display': 'none'}
        ),

    ])

app = dash.Dash()
app.layout = serve_layout()





@app.callback(
    Output('hidden-data', 'children'),
    [Input('puzzle-selector', 'value'),
     Input('heatmap', 'clickData'),
     Input('button', 'n_clicks'),
     Input('button-clear', 'n_clicks')],
    [State('hidden-data', 'children')])
def update_data(dropdown_value, clickData, n_clicks, n_clicks_clear, game_json):

    game_dict = json.loads(game_json)
    
    # Trigger 1: If puzzle has changed, reload all game data
    if dropdown_value != game_dict['puzzle']:
        
        game = GridMap(
            sp.nono_dict[dropdown_value]['col_rules'], 
            sp.nono_dict[dropdown_value]['row_rules']
        )

        game_dict['col_rules'] = game._col_rules
        game_dict['row_rules'] = game._row_rules
        game_dict['row_labels'] = game._row_labels
        game_dict['col_labels'] = game._col_labels
        game_dict['w'] = game._w
        game_dict['h'] = game._h
        game_dict['grid'] = game._grid.tolist()
        game_dict['puzzle'] = dropdown_value
        game_dict['n_clicks'] = 0
        game_dict['n_clicks_clear'] = 0

    # Trigger 2: Solve button clicked
    elif (n_clicks is not None) and (n_clicks > game_dict['n_clicks']):
        print("Solve problem")

        game = GridMap(
            sp.nono_dict[dropdown_value]['col_rules'], 
            sp.nono_dict[dropdown_value]['row_rules']
        )
        game._grid = np.array(game_dict['grid'])
        
        # Solve problem
        game = opt.rule_based_simplify(game)
        model = opt.opt_model(game)

        pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

        game._grid = opt.solve_model(model, game._h, game._w)

        game_dict['grid'] = game._grid.tolist()

        game_dict['n_clicks'] = n_clicks

    # Clear button clicked
    if (n_clicks_clear is not None) and \
        (n_clicks_clear > game_dict['n_clicks_clear']):

        game_dict['grid'] = np.zeros((game_dict['h'], game_dict['w'])).tolist()
        game_dict['n_clicks_clear'] = n_clicks_clear


    # Trigger 3: if user clicked on the grid
    elif clickData is not None:
        grid = np.array(game_dict['grid'])
        
        x = clickData["points"][0]["x"]
        y = clickData["points"][0]["y"]
        z = clickData["points"][0]["z"] 

        grid[y, x] = game_dict['changes'][str(z)]
        game_dict['grid'] = grid.tolist()

    # print('before return', game_dict['grid'])
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

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)