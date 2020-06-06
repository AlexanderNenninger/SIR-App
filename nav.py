import dash_bootstrap_components as dbc


def navbar():
    # Make top nav bar
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("SIR", href='/sir')),
        ],
        color='light'
    )
    return navbar
