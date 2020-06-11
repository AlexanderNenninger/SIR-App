import numpy as np


def eval_on_grid_3d(func: callable, x_min: np.ndarray, x_max: np.ndarray, t: float,
                    n_points: int, *args, **kwargs) -> (np.ndarray, np.ndarray):
    x = np.linspace(x_min[0], x_max[0], n_points)
    y = np.linspace(x_min[1], x_max[1], n_points)
    z = np.linspace(x_min[2], x_max[2], n_points)

    xx = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    f_map = [func(t, x, *args, **kwargs) for x in xx]
    u = np.array(f_map)

    return xx, u


def eval_on_grid_2d(func: callable, x_min: np.ndarray, x_max: np.ndarray, t: float,
                    n_points: int, *args, **kwargs) -> (np.ndarray, np.ndarray):
    x = np.linspace(x_min[0], x_max[0], n_points)
    y = np.linspace(x_min[1], x_max[1], n_points)
    
    xx = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    f_map = [func(t, x, *args, **kwargs) for x in xx]
    u = np.array(f_map)

    return xx, u
