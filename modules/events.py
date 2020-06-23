class event:
    def __init__(self, func, terminal=False, direction=0):
        self.func = func
        self.terminal = terminal
        self.direction = direction

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)