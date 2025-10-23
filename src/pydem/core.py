import taichi as ti
from .config import ndarray_1d_float, ndarray_1d_int, ndarray_2d_float, vec2_float

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
