import dash_html_components as html


description_SIR = html.Div([
    html.H3('The SIR Infectious Disease Model'),
    html.P("The ODE this model is based on reads"),
    html.P(
        children=r"""
\begin{align}
    \dot{S} &= - \beta I S + \mu (I + R)\\
    \dot{I} &= \beta I S - (\gamma + \mu) I\\
    \dot{R} &= \gamma I - \mu R
\end{align}
    """),
    html.P(r"""where S denotes the number of susceptible, I the number of infected and R the number of removed people."""),
    html.P("Parameters:"),
    html.Ul([
        html.Li(r"Infection rate $\beta$"),
        html.Li(r"Recovery rate $\gamma$"),
        html.Li(r"Birth/Death rate $\mu$")
    ])

], style={'margin': '5px'})

description_SIS = html.Div([
    html.H3('The SIS Infectious Disease Model'),
    html.P("The ODE this model is based on reads"),
    html.P(
        children=r"""
\begin{align}
    \dot{S} &= - \beta \frac{SI}{N} + \gamma I + (\delta - p \delta)N - \sigma S\\
    \dot{I} &= \beta  \frac{SI}{N} - \gamma I + p \delta N - (\sigma + \epsilon) I
\end{align}
    """),
    html.P(r"""where S denotes the number of susceptible and I the number of infected people""")

], style={'margin': '5px'})