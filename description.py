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
    html.P(r"""where S denotes the number of infected, I the number of infected and R the number of removed people.""")

], style={'margin': '5px'})

description_SIS = html.Div([
    html.H3('The SIS Infectious Disease Model'),
    html.P("The ODE this model is based on reads"),
    html.P(
        children=r"""
\begin{align}
    \dot{S} &= - \beta I S + \mu (I + R)\\
    \dot{I} &= TEST
\end{align}
    """),
    html.P(r"""where S denotes the number of infected, I the number of infected and R the number of removed people.""")

], style={'margin': '5px'})