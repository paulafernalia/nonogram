import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import json
import sample_puzzle
from utils import GridMap

# Colour definitions
b = 1
w = 0
g = 0.2


game = GridMap(
    sample_puzzle.col_rules, 
    sample_puzzle.row_rules
    )

print(game.grid)

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(
        id='heatmap',
        figure=go.Figure(
            data=[
                go.Heatmap(
                    x= [0,1,2],
                    y=[0,1,2],
                    z = [[g, b, g],[w, b, g]],
                    colorscale = 'Greys',
                    reversescale = True,
                    showscale = False,
                    xgap=5,
                    ygap=5,
                )
            ],
            layout = 
                go.Layout(
                    title='Nonogram',
                    height=500,
                    width=1000,
                    plot_bgcolor=('rgb(255,255,255)'),
                    yaxis=dict(
                        ),
                    xaxis=dict(
                        title='Test'
                        )
                ),
        ),
    ),
    html.Div(id='output')
])


@app.callback(
    Output('output', 'children'),
    [Input('heatmap', 'clickData')])
def display_hoverdata(clickData):

    if clickData is not None:

        return [
            clickData["points"][0]["z"]
        ]


# if __name__ == '__main__':
#     app.run_server(debug=True)