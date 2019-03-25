import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input

app = dash.Dash()


app.layout = html.Div([
    html.Button(id='button', children='Button'),
    html.Br(),
    dcc.Dropdown(id='dropdown',
                 options=[{'value': True, 'label': 'True'},
                          {'value': False, 'label': 'False'}])
])

@app.callback(Output('button', 'disabled'),
             [Input('dropdown', 'value')])
def set_button_enabled_state(on_off):
    return on_off


if __name__ == '__main__':
    app.run_server()