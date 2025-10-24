import taichi as ti
import numpy as np
from .config import ndarray_1d_float, ndarray_2d_float, vec2_float, dtype_float, dtype_float_np, dtype_int
from .neighbors import build_neighbors_naive
from .box import dist_pbc
from .potential import soft_disk
from .core import compute_forces, scale_to_temperature
from .general import any_exceeds
from .integrator import update_positions, update_velocities

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
    flag = ti.field(dtype=dtype_int, shape=())
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