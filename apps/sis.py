import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.integrate import solve_ivp
import numpy as np

from application import app
from collapse import collapse
from modules.events import event
from modules.eval_on_grid import eval_on_grid_2d
from description import description_SIS


# RHS of ODE
def f_sis(t, y, beta, gamma, delta, sigma, epsilon, p):
    S = y[0]
    I = y[1]
    N = S + I
    return np.array([
        - beta * S * I / N + gamma * I + delta * (1 - p * I / N) * N - sigma * S,
        beta * S * I / N - gamma * I + p * delta * I - (sigma + epsilon) * I
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
            dbc.Input(id='inp-S0-sis', type='number', step=1, placeholder='Initial Susceptible')
        ], width=2),
        dbc.Col([
            dbc.Input(id='inp-I0-sis', type='number', step=1, placeholder='Initial Infected')
        ], width=2),
        dbc.Col([
            dcc.Markdown(children='R_0 =', id="mkd_r0")
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
                    dbc.Col([
                        html.Hr(),
                    ]),
                ]),
                # Population Parameters
                dbc.Row([
                    dbc.Col('Delta', width=3),
                    dbc.Col([
                        dcc.Slider(id='delta', min=0, max=.1, step=.001, value=.052, marks={0:'0', .1:'0.1'}),
                    ], width=9),
                ]),
                dbc.Row([
                    dbc.Col('Sigma', width=3),
                    dbc.Col([
                        dcc.Slider(id='sigma', min=0, max=.1, step=.001, value=.05, marks={0:'0', .1:'0.1'}),
                    ], width=9),
                ]),
                dbc.Row([
                    dbc.Col('Epsilon', width=3),
                    dbc.Col([
                        dcc.Slider(id='epsilon', min=0, max=.05, step=.001, value=0.002, marks={0:'0', .05:'0.05'}),
                    ], width=9),
                ]),
                dbc.Row([
                    dbc.Col('P', width=3),
                    dbc.Col([
                        dcc.Slider(id='p', min=0, max=1, step=.01, value=0.01, marks={0:'0', 1:'1'}),
                    ], width=9),
                ]),
            ),
        ],),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            description_SIS
        ], width=6),
        dbc.Col([
            dcc.Graph(id='2d-path', style={'height': '60vh'}),
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='time-series-graph-sis'),
        ], width=12),
    ]),
    html.Div(id='dummy-div'),
], className='sir')


@app.callback(
    Output('mkd_r0', 'children'),
    [
        Input('beta', 'value'),
        Input('gamma', 'value'),
        Input('delta', 'value'),
        Input('sigma', 'value'),
        Input('epsilon', 'value'),
        Input('p', 'value'),
    ]
)
def r0_sis(beta, gamma, delta, sigma, epsilon, p):
    try:
        R_0 = beta * delta / (sigma * (gamma + sigma + epsilon - p * delta))
        return "R_0 = {:.2f}".format(R_0)
    except ZeroDivisionError:
        return "R_0 = undef."

@app.callback(
    Output('2d-path', 'figure'),
    [
        Input('beta', 'value'),
        Input('gamma', 'value'),
        Input('delta', 'value'),
        Input('sigma', 'value'),
        Input('epsilon', 'value'),
        Input('p', 'value'),
        Input('inp-S0-sis', 'value'),
        Input('inp-I0-sis', 'value'),
        Input('dummy-div', 'children')
    ],
)
def update_2d_sis(beta, gamma, delta, sigma, epsilon, p, S_0, I_0, aux):
    '''Updates the 3d Plot'''
    # Test values
    T = 200
    # Handle None Case
    S_0 = S_0 or 80
    I_0 = I_0 or 20

    # Data for trajectory
    t, y = SIS(S_0, I_0, beta, gamma, delta, sigma, epsilon, p, T)

    # Draw vector field
    S_max = y[0].max()
    I_max = y[1].max()
    grid_max = np.array([S_max, I_max])
    xx, uu = eval_on_grid_2d(
        func=f_sis,
        x_min=np.zeros_like(grid_max),
        x_max=grid_max,
        t=0,
        n_points=20,
        beta=beta,
        gamma=gamma,
        delta=delta,
        sigma=sigma,
        epsilon=epsilon,
        p=p
    )
    fig = ff.create_quiver(
        x=xx[:, 0],
        y=xx[:, 1],
        u=uu[:, 0],
        v=uu[:, 1],
        name='Vector Field',
        scale=.4,
    )

    # Make figure
    fig.add_traces(
        [
            go.Scatter(
                name='Intial Conditions',
                x=[S_0],
                y=[I_0],
                mode='markers',
                marker=dict(size=10, opacity=.5),
            ),
            go.Scatter(
                name='Sample Trajectory',
                x=y[0],
                y=y[1],
                mode='lines+markers',
                marker=dict(
                    color=t,
                    colorscale='Viridis',
                    size=2,
                ),
            ),
        ]
    )

    fig.update_layout(
        title="Trajectories of the SIS Model",
        xaxis_title='Susceptible',
        yaxis_title='Infected',
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255,255,255,.5)',
        ),
    )
    return fig


@app.callback(
    Output('time-series-graph-sis', 'figure'),
    [
        Input('beta', 'value'),
        Input('gamma', 'value'),
        Input('delta', 'value'),
        Input('sigma', 'value'),
        Input('epsilon', 'value'),
        Input('p', 'value'),
        Input('inp-S0-sis', 'value'),
        Input('inp-I0-sis', 'value'),
        Input('dummy-div', 'children')
    ],
)
def update_timeseries_sis(beta, gamma, delta, sigma, epsilon, p, S_0, I_0, aux):
    '''Updates the Time Series'''
    T = 200
    #
    S_0 = S_0 or 80
    I_0 = I_0 or 20
    t, y = SIS(S_0, I_0, beta, gamma, delta, sigma, epsilon, p, T)

    fig = go.Figure(
        data=[
            go.Scatter(x=t, y=y[0], name='Susceptible'),
            go.Scatter(x=t, y=y[1], name='Infected'),
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255,255,255,.5)',
        ),
    )
    return fig
