import taichi as ti
from .config import dtype_float, vec2_float

@ti.func
def dist(r_i: vec2_float, r_j: vec2_float, box_size: vec2_float) -> vec2_float:
    return r_i - r_j

@ti.func
def dist_pbc(r_i: vec2_float, r_j: vec2_float, box_size: vec2_float) -> vec2_float:
    r_ij = r_i - r_j
    r_ij -= box_size * ti.round(r_ij / box_size)
    return r_ij