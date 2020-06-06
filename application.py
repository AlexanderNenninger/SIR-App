import dash
from dash_bootstrap_components.themes import BOOTSTRAP

app = dash.Dash(
    __name__,
    external_stylesheets=[BOOTSTRAP],
    suppress_callback_exceptions=True
)
application = app.server
app.title = "Disease Models"