import dash
from dash_bootstrap_components.themes import BOOTSTRAP

mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'

app = dash.Dash(
    __name__,
    external_stylesheets=[BOOTSTRAP],
    external_scripts=[mathjax],
    suppress_callback_exceptions=True
)
application = app.server
app.title = "Disease Models"
