import taichi as ti
import numpy as np
from .config import ndarray_1d_float, ndarray_2d_float, vec2_float, dtype_float, dtype_float_np, dtype_int
from .neighbors import build_neighbors_naive
from .box import dist_pbc
from .potential import soft_disk
from .core import compute_forces

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

flag = ti.field(dtype=dtype_int, shape=())

@ti.kernel
def any_exceeds(a: ndarray_1d_float, thresh: float, out: ti.template()):
    out[None] = 0
    N = ti.cast(a.shape[0], ti.int32)
    for i in range(N):
        if a[i] > thresh:
            ti.atomic_max(out[None], 1)

@ti.kernel
def reduce_sum_vel(vel: ndarray_2d_float, out: ti.template()):
    out[None] = ti.Vector.zero(dtype_float, 2)
    N = ti.cast(vel.shape[0], ti.int32)
    for i in range(N):
        out[None] += vel[i]

@ti.kernel
def subtract_mean(vel: ndarray_2d_float, mean: vec2_float):
    N = ti.cast(vel.shape[0], ti.int32)
    for i in range(N):
        vel[i] = vel[i] - mean

@ti.kernel
def reduce_ke_sum(vel: ndarray_2d_float, mass: ndarray_1d_float, out: ti.template()):
    out[None] = ti.cast(0.0, dtype_float)
    N = ti.cast(vel.shape[0], ti.int32)
    for i in range(N):
        v = vel[i]
        out[None] += 0.5 * mass[i] * v.dot(v)

@ti.kernel
def scale_velocities(vel: ndarray_2d_float, s: dtype_float):
    N = ti.cast(vel.shape[0], ti.int32)
    for i in range(N):
        vel[i] = vel[i] * s

@ti.kernel
def compute_ke(
    vel: ndarray_2d_float,
    mass: ndarray_1d_float,
    ke: ndarray_1d_float):
    N = vel.shape[0]
    for i in range(N):
        v = vel[i]
        ke[i] = 0.5 * mass[i] * v.dot(v)

def compute_temperature(vel: ndarray_2d_float,
                        mass: ndarray_1d_float) -> float:
    ke_sum = ti.field(dtype=dtype_float, shape=())
    reduce_ke_sum(vel, mass, ke_sum)
    N = vel.shape[0]
    dof = 2 * N  # in 2D; adjust to 2*N - 2 if you remove COM below
    return float(2.0 * ke_sum[None] / dof)

def scale_to_temperature(T: float,
                         vel: ndarray_2d_float,
                         mass: ndarray_1d_float):
    v_sum = ti.Vector.field(2, dtype=dtype_float, shape=())
    reduce_sum_vel(vel, v_sum)
    N = vel.shape[0]
    mean = v_sum[None] / N
    subtract_mean(vel, mean)

    ke_sum = ti.field(dtype=dtype_float, shape=())
    reduce_ke_sum(vel, mass, ke_sum)
    dof = 2 * N  # use 2*N - 2 if COM is constrained
    KE_target = 0.5 * dof * T
    s = (KE_target / float(ke_sum[None])) ** 0.5
    scale_velocities(vel, s)

if __name__ == "__main__":
    N = 1000
    phi = 0.5
    e_c = 1.0
    temp = 1e-4
    N_steps = 1000
    dt = 1e-2
    verlet_skin = 0.5

    radii_np = np.ones(N, dtype=dtype_float_np) * 0.5
    radii_np[N // 2:] *= 1.4
    radii = ti.ndarray(dtype=dtype_float, shape=(N,))
    radii.from_numpy(radii_np)

    mass_np = np.ones(N, dtype=dtype_float_np)
    mass = ti.ndarray(dtype=dtype_float, shape=(N,))
    mass.from_numpy(mass_np)

    box_size_np = np.ones(2, dtype=dtype_float_np) * np.sqrt(np.sum(np.pi * radii_np ** 2) / phi)
    box_size = vec2_float(box_size_np[0], box_size_np[1])

    pos_np = np.random.uniform(low=0, high=1, size=(N, 2)).astype(dtype_float_np)
    pos_np *= box_size_np
    pos = ti.ndarray(dtype=vec2_float, shape=(N,))
    pos.from_numpy(pos_np)

    vel_np = np.random.normal(loc=0, scale=np.sqrt(temp), size=(N, 2)).astype(dtype_float_np)
    vel_np -= np.mean(vel_np, axis=0)
    scale = np.sqrt(temp / 1)
    vel = ti.ndarray(dtype=vec2_float, shape=(N,))
    vel.from_numpy(vel_np)
    scale_to_temperature(temp, vel, mass)

    force = ti.ndarray(dtype=vec2_float, shape=(N,))
    last_pos = ti.ndarray(dtype=vec2_float, shape=(N,))
    disp_sq = ti.ndarray(dtype=dtype_float, shape=(N,))
    pe = ti.ndarray(dtype=dtype_float, shape=(N,))
    ke = ti.ndarray(dtype=dtype_float, shape=(N,))

    verlet_rad = (1 + verlet_skin) * 2 * np.max(radii_np)
    verlet_thresh_sq = (verlet_skin / 2.0) ** 2
    neigh_ids, neigh_offset = build_neighbors_naive(pos, verlet_rad, dist_pbc, box_size)
    last_pos.copy_from(pos)
    disp_sq.fill(0)

    for i in range(N_steps):
        update_velocities(vel, force, mass, dt)
        update_positions(pos, last_pos, vel, disp_sq, dt)
        force.fill(0)
        pe.fill(0)
        compute_forces(pos, force, pe, radii, neigh_ids, neigh_offset, dist_pbc, box_size, soft_disk, e_c)
        update_velocities(vel, force, mass, dt)

        # update neighbors if needed
        any_exceeds(disp_sq, verlet_thresh_sq, flag)
        if bool(flag[None]):
            neigh_ids, neigh_offset = build_neighbors_naive(pos, verlet_rad, dist_pbc, box_size)
            last_pos.copy_from(pos)
            disp_sq.fill(0)