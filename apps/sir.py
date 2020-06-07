import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import numpy as np

from application import app
from nav import navbar
from collapse import collapse
from modules.events import event

stop_when_over = event(lambda t, y, beta, gamma: y[1] - .005, terminal=True, direction=-1)


def f_sir(t, y, beta, gamma):
    S = y[0]
    I = y[1]
    return np.array([
        - beta * I * S,
        beta * I * S - gamma * I,
        gamma * I
    ])


def SIR(S_0: float, I_0: float, R_0: float, beta: float,
        gamma: float,  T: float) -> np.ndarray:
    """Computes Solution to SIR Model from t=0 to T
    S_0, I_0, R_0 <float>: initial values
    beta<float>: Infection Rate * Interaction Rate
    gamma<float>: Removal Rate
    T<float>: Max Time
    returns <np.ndarray>(3,1000):
    """
    t_eval = np.linspace(0, T, 1000)
    y0 = np.array([
        S_0,
        I_0,
        R_0
    ])
    N = y0.sum()
    y0 = y0 / N

    sol = solve_ivp(
        f_sir,
        t_span=[0, T],
        y0=y0,
        method='Radau',
        t_eval=t_eval,
        args=(beta, gamma),
        events=[stop_when_over]
    )
    if sol.success:
        return sol.t, sol.y * N
    else:
        return np.array([0]), y0 * N


layout = html.Div([
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Input(id='inp-S0', type='number', step=1, placeholder='Initial Susceptible')
        ], width=2),
        dbc.Col([
            dbc.Input(id='inp-I0', type='number', step=1, placeholder='Initial Infected')
        ], width=2),
        dbc.Col([
            dbc.Input(id='inp-R0', type='number', step=1, placeholder='Initial Removed')
        ], width=2),
        dbc.Col([
            collapse(
                dbc.Row([
                    dbc.Col('Beta', width=3),
                    dbc.Col([
                        dcc.Slider(id='beta', min=0, max=1, step=.01, value=.5),
                    ], width=9),
                ]),
                dbc.Row([
                    dbc.Col('Gamma', width=3),
                    dbc.Col([
                        dcc.Slider(id='gamma', min=0, max=1, step=.01, value=.2),
                    ], width=9),
                ])
            ),
        ],),
    ],),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='3d-path'),
        ], width=4),
        dbc.Col([
            dcc.Graph(id='time-series-graph'),
        ], width=8),
    ]),
    html.Div(id='dummy-div')
], className='sir')


@app.callback(
    Output('3d-path', 'figure'),
    [
        Input('beta', 'value'),
        Input('gamma', 'value'),
        Input('inp-S0', 'value'),
        Input('inp-I0', 'value'),
        Input('inp-R0', 'value'),
        Input('dummy-div', 'children')  
    ],
)
def update_3d(beta, gamma, S_0, I_0, R_0, aux):
    '''Updates the 3d Plot'''
    # Test values
    T = 1000
    #
    S_0 = S_0 or 100
    I_0 = I_0 or 1
    R_0 = R_0 or 0
    N = S_0 + I_0 + R_0
    t, y = SIR(S_0, I_0, R_0, beta, gamma, T)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=y[0],
        y=y[1],
        z=y[2],
        mode='lines+markers',
        marker=dict(
            color=t,
            colorscale='Viridis',
            size=2,
        ),
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            aspectratio=dict(x=1, y=1, z=1),
            xaxis_title='Susceptible',
            yaxis_title='Infected',
            zaxis_title='Removed',

            xaxis=dict(range=[0, N]),
            yaxis=dict(range=[0, N]),
            zaxis=dict(range=[0, N]),
        ),
    )
    return fig


@app.callback(
    Output('time-series-graph', 'figure'),
    [
        Input('beta', 'value'),
        Input('gamma', 'value'),
        Input('inp-S0', 'value'),
        Input('inp-I0', 'value'),
        Input('inp-R0', 'value'),
        Input('dummy-div', 'children')  
    ],
)
def update_timeseries(beta, gamma, S_0, I_0, R_0, aux):
    '''Updates the Time Series'''
    T = 1000
    #
    S_0 = S_0 or 100
    I_0 = I_0 or 1
    R_0 = R_0 or 0
    N = S_0 + I_0 + R_0
    t, y = SIR(S_0, I_0, R_0, beta, gamma, T)

    fig = go.Figure(
        data=[
            go.Scatter(x=t, y=y[0], name='Susceptible'),
            go.Scatter(x=t, y=y[1], name='Infected'),
            go.Scatter(x=t, y=y[2], name='Removed'),
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig