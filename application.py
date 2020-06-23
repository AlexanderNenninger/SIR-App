import flask
import os

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import application, app

from apps import sir
from apps import sis
from nav import navbar

STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar(),
    html.Div(id='page-content', style={'margin': '5px'}),
])


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'), ]
)
def display_page(pathname):
    if pathname in ['/', '']:
        return sir.layout
    elif pathname in ['/sir', '/sir/']:
        return sir.layout
    elif pathname in ['/sis', '/sis/']:
        return sis.layout
    else:
        return '404'

@app.server.route('/static/<resource>')
def serve_static(resource):
    return flask.send_from_directory(STATIC_PATH, resource)

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=8000)
