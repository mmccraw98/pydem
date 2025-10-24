import taichi as ti
from .config import ndarray_1d_float, ndarray_1d_int, ndarray_2d_float, vec2_float, dtype_float

@ti.kernel
def compute_forces(
    pos: ndarray_2d_float,
    force: ndarray_2d_float,
    pe: ndarray_1d_float,
    radii: ndarray_1d_float,
    neighbor_ids: ndarray_1d_int,
    neighbor_offset: ndarray_1d_int,
    dist_func: ti.template(),
    box_size: vec2_float,
    potential_func: ti.template(),
    e_c: float):
    N = pos.shape[0]
    for i in range(N):
        pos_i = pos[i]
        rad_i = radii[i]
        force_i = force[i]
        pe_i = pe[i]
        for _j in range(neighbor_offset[i], neighbor_offset[i + 1]):
            j = neighbor_ids[_j]
            if i == j:
                continue
            sigma_ij = rad_i + radii[j]
            r_ij = dist_func(pos_i, pos[j], box_size)
            pot_ij = potential_func(r_ij, sigma_ij, e_c)
            force_i += pot_ij.f
            pe_i += pot_ij.pe * 0.5  # avoid double counting
        force[i] = force_i
        pe[i] = pe_i

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