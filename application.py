import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import application, app

from apps import sir
from apps import sis
from nav import navbar

from apps import sir
from apps import sis
from nav import navbar

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


if __name__ == "__main__":
    application.run()
