import taichi as ti
from .config import ndarray_2d_float, ndarray_1d_float

@ti.kernel
def update_velocities(
    vel: ndarray_2d_float,
    force: ndarray_2d_float,
    mass: ndarray_1d_float,
    dt: float):
    N = vel.shape[0]
    for i in range(N):
        vel[i] += force[i] * dt / (2 * mass[i])

@ti.kernel
def update_positions(
    pos: ndarray_2d_float,
    last_pos: ndarray_2d_float,
    vel: ndarray_2d_float,
    disp_sq: ndarray_1d_float,
    dt: float):
    N = vel.shape[0]
    for i in range(N):
        pos[i] += vel[i] * dt
        disp_vec = pos[i] - last_pos[i]
        disp_sq[i] = disp_vec.dot(disp_vec)