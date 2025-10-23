import taichi as ti
from .config import vec2_float, dtype_float

pair_potential = ti.types.struct(f=vec2_float, pe=dtype_float)

@ti.func
def soft_disk(
    r_ij: vec2_float,
    sigma_ij: float,
    e_c: float) -> pair_potential:
    d_ij_sq = r_ij.dot(r_ij)
    d_ij = ti.sqrt(d_ij_sq)
    inv_d_ij = 1.0 / d_ij
    overlap = sigma_ij - d_ij
    mask = ti.cast(d_ij_sq < sigma_ij * sigma_ij, dtype_float)
    f = e_c * overlap * inv_d_ij * r_ij * mask
    pe = 0.5 * e_c * overlap * overlap * mask
    return pair_potential(f=f, pe=pe)