import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from application import app, application

from apps import sir

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'), ]
)
def display_page(pathname):
    if pathname == '/' or '':
        return sir.layout
    elif pathname == '/SIR' or '/sir':
        return sir.layout
    else:
        return '404'


if __name__ == "__main__":
    application.run(debug=True, host='0.0.0.0', port=8080)
