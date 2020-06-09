import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import numpy as np

from collapse import collapse
from modules.events import event
from description import description_SIS


# RHS of ODE
def f_sis(t, y, beta, gamma, delta, sigma, epsilon, p):
    S = y[0]
    I = y[1]
    N = S + I
    return np.array([
        gamma * N + beta * S * I / N - (p * gamma - delta) * I - sigma * S,
        beta * S * I / N - (delta + sigma + epsilon - p * gamma) * I
    ])


# Stop at equilibrium
stop_when_over = event(
    lambda t, y, beta, gamma, delta, sigma, epsilon, p: np.linalg.norm(f_sis(t, y, beta, gamma, delta, sigma, epsilon, p)) - .0001, 
    terminal=True, 
    direction=-1
)

# Model
def SIS(S_0: float, I_0: float, beta: float, gamma: float, delta: float,
        sigma: float, epsilon, p, T):
    """Computes Solution to SIS Model from t=0 to T
    S_0, I_0 <float>
    returns <np.ndarray>(3,1000)
    """
    t_eval = np.linspace(0, T, 1000)
    y0 = np.array([
        S_0,
        I_0,
    ])
    sol = solve_ivp(
        f_sis,
        t_span=[0, T],
        y0=y0,
        method='Radau',
        t_eval=t_eval,
        args=(beta, gamma, delta, sigma, epsilon, p),
        events=[stop_when_over]
    )
    if sol.success:
        return sol.t, sol.y
    else:
        return np.array([0]), np.array([y0])

layout = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Input(id='inp-S0', type='number', step=1, placeholder='Initial Susceptible')
        ], width=2),
        dbc.Col([
            dbc.Input(id='inp-I0', type='number', step=1, placeholder='Initial Infected')
        ], width=2),
        dbc.Col([
            collapse(
                dbc.Row([
                    dbc.Col('Beta', width=3),
                    dbc.Col([
                        dcc.Slider(id='beta', min=0, max=1, step=.01, value=.5, marks={0:'0', 1:'1'}),
                    ], width=9),
                ]),
                dbc.Row([
                    dbc.Col('Gamma', width=3),
                    dbc.Col([
                        dcc.Slider(id='gamma', min=0, max=1, step=.01, value=.2, marks={0:'0', 1:'1'}),
                    ], width=9),
                ]),
                dbc.Row([
                    dbc.Col('Delta', width=3),
                    dbc.Col([
                        dcc.Slider(id='delta', min=0, max=1, step=.01, value=.1, marks={0:'0', 1:'1'}),
                    ], width=9),
                ]),
                dbc.Row([
                    dbc.Col('Sigma', width=3),
                    dbc.Col([
                        dcc.Slider(id='sigma', min=0, max=1, step=.01, value=.1, marks={0:'0', 1:'1'}),
                    ], width=9),
                ]),
                dbc.Row([
                    dbc.Col('Epsilon', width=3),
                    dbc.Col([
                        dcc.Slider(id='epsilon', min=0, max=1, step=.01, value=0., marks={0:'0', 1:'1'}),
                    ], width=9),
                ]),
                dbc.Row([
                    dbc.Col('P', width=3),
                    dbc.Col([
                        dcc.Slider(id='p', min=0, max=1, step=.01, value=0., marks={0:'0', 1:'1'}),
                    ], width=9),
                ]),                                                
            ),
        ],),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            description_SIS
        ], width=4),
        dbc.Col([
            dcc.Graph(id='2d-path', style={'height': '90vh'}),
        ], width=8),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='time-series-graph'),
        ], width=12),
    ]),
    html.Div(id='dummy-div'),
], className='sir')