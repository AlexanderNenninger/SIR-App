import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import numpy as np

from app import app
from collapse import collapse
from modules.events import event
from modules.eval_on_grid import eval_on_grid_3d
from description import description_SIR


def f_sir(t, y, beta, gamma, mu):
    S = y[0]
    I = y[1]
    R = y[2]
    return np.array([
        - beta * I * S + mu * (I + R),
        beta * I * S - (gamma + mu) * I,
        gamma * I - mu * R
    ])


stop_when_over = event(lambda t, y, beta, gamma, mu: np.linalg.norm(f_sir(t, y, beta, gamma, mu)) - .0001, terminal=True, direction=-1)


def SIR(S_0: float, I_0: float, R_0: float, beta: float,
        gamma: float, mu: float,  T: float) -> np.ndarray:
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
        args=(beta, gamma, mu),
        events=[stop_when_over]
    )
    if sol.success:
        return sol.t, sol.y * N
    else:
        return np.array([0]), np.array([y0 * N])


layout = html.Div([
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
                    dbc.Col('Mu', width=3),
                    dbc.Col([
                        dcc.Slider(id='mu', min=0, max=.2, step=.0001, value=0., marks={0:'0', .2:'0.2'}),
                    ], width=9),
                ]),
            ),
        ],),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            description_SIR
        ], width=6),
        dbc.Col([
            dcc.Graph(id='3d-path', style={'height': '80vh'}),
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='time-series-graph'),
        ], width=12),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H3("Lyapunov function of the SIR-Model")
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Embed(type="text/html", src="/static/sir_lyapunov.html", width="900", height="600"),
        ], width=12),
    ]),
    html.Div(id='dummy-div'),
], className='sir')


@app.callback(
    Output('3d-path', 'figure'),
    [
        Input('beta', 'value'),
        Input('gamma', 'value'),
        Input('mu', 'value'),
        Input('inp-S0', 'value'),
        Input('inp-I0', 'value'),
        Input('inp-R0', 'value'),
        Input('dummy-div', 'children')  
    ],
)
def update_3d(beta, gamma, mu, S_0, I_0, R_0, aux):
    '''Updates the 3d Plot'''
    # Test values
    T = 1000
    # Handle None Case
    if S_0 is None: S_0 = 80
    if I_0 is None: I_0 = 20
    if R_0 is None: R_0 = 20
    N = S_0 + I_0 + R_0

    # Data for trajectory
    t, y = SIR(S_0, I_0, R_0, beta, gamma, mu, T)

    # Data for Cone Plot
    x, u = eval_on_grid_3d(
        func=f_sir,
        x_min=np.zeros(3),
        x_max=N*np.ones(3),
        t=0,
        n_points=10,
        beta=beta,
        gamma=gamma,
        mu=mu,
    )
    # Make figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
                name='Intial Conditions',
                x=[S_0],
                y=[I_0],
                z=[R_0],
                mode='markers',
                marker=dict(size=10),
            ),
            go.Scatter3d(
                name='Sample Trajectory',
                x=y[0],
                y=y[1],
                z=y[2],
                mode='lines+markers',
                marker=dict(
                    color=t,
                    colorscale='Viridis',
                    size=2,
                ),  
            ),
            go.Cone(
                name='Vector Field',
                opacity=.6,
                x=x[:, 0],
                y=x[:, 1],
                z=x[:, 2],
                u=u[:, 0],
                v=u[:, 1],
                w=u[:, 2],
                autocolorscale=True,
                showscale=False,
                sizemode='absolute',
            )
        ]
    )

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
        legend=dict(x=0, y=1),
    )
    return fig


@app.callback(
    Output('time-series-graph', 'figure'),
    [
        Input('beta', 'value'),
        Input('gamma', 'value'),
        Input('mu', 'value'),
        Input('inp-S0', 'value'),
        Input('inp-I0', 'value'),
        Input('inp-R0', 'value'),
        Input('dummy-div', 'children')  
    ],
)
def update_timeseries(beta, gamma, mu, S_0, I_0, R_0, aux):
    '''Updates the Time Series'''
    T = 1000
    #
    if S_0 is None: S_0 = 80
    if I_0 is None: I_0 = 20
    if R_0 is None: R_0 = 20
    t, y = SIR(S_0, I_0, R_0, beta, gamma, mu, T)

    fig = go.Figure(
        data=[
            go.Scatter(x=t, y=y[0], name='Susceptible'),
            go.Scatter(x=t, y=y[1], name='Infected'),
            go.Scatter(x=t, y=y[2], name='Removed'),
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0, y=1),
    )
    return fig
