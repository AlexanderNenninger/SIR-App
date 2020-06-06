import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from application import app


def collapse(*collapse_fields):
    # Define collapse group
    collapse = html.Div([
        dbc.Button(
            "More Parameters",
            id="collapse-button",
            color='primary',
            block=True,
        ),
        dbc.Collapse(
            children=[
                dbc.Card(dbc.CardBody(
                    children=[field for field in collapse_fields]
                ))
            ], id='collapse',),
    ])
    return collapse


@app.callback(
    Output('collapse', 'is_open'),
    [Input('collapse-button', 'n_clicks')],
    [State('collapse', 'is_open')]
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open