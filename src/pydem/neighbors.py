import taichi as ti

from .config import dtype_int, dtype_float, ndarray_2d_float, ndarray_1d_int, vec2_float, vec2_int
from .general import sum1d_int, exclusive_scan_int

def init_neighbor_arrays(
    neigh_count: ndarray_1d_int) -> (ndarray_1d_int, ndarray_1d_int):
    N = neigh_count.shape[0]
    neigh_offset = ti.ndarray(dtype=dtype_int, shape=(N + 1,))
    tmp = ti.ndarray(dtype=dtype_int, shape=(N + 1,))
    exclusive_scan_int(neigh_count, neigh_offset, tmp)
    total = int(neigh_offset.to_numpy()[-1])
    neigh_ids = ti.ndarray(dtype=dtype_int, shape=(total,))
    return neigh_ids, neigh_offset

@ti.kernel
def count_neighbors_naive(
    pos: ndarray_2d_float,
    neigh_count: ndarray_1d_int,
    neigh_rad: float,
    dist_func: ti.template(),
    box_size: vec2_float):
    N = pos.shape[0]
    neigh_rad_sq = neigh_rad ** 2
    for i in range(N):
        num_neighbors = 0
        pos_i = pos[i]
        for j in range(N):
            if i == j:
                continue
            r_ij = dist_func(pos_i, pos[j], box_size)
            if r_ij.dot(r_ij) < neigh_rad_sq:
                num_neighbors += 1
        neigh_count[i] = num_neighbors

@ti.kernel
def assign_neighbor_ids_naive(
    pos: ndarray_2d_float,
    neigh_ids: ndarray_1d_int,
    neigh_offset: ndarray_1d_int,
    neigh_rad: float,
    dist_func: ti.template(),
    box_size: vec2_float):
    N = pos.shape[0]
    neigh_rad_sq = neigh_rad ** 2
    for i in range(N):
        neigh_idx = neigh_offset[i]
        pos_i = pos[i]
        for j in range(N):
            if i == j:
                continue
            r_ij = dist_func(pos_i, pos[j], box_size)
            if r_ij.dot(r_ij) < neigh_rad_sq:
                neigh_ids[neigh_idx] = j
                neigh_idx += 1

def build_neighbors_naive(
    pos: ndarray_2d_float,
    neigh_rad: float,
    dist_func: ti.template(),
    box_size: vec2_float) -> (ndarray_1d_int, ndarray_1d_int):
    N = pos.shape[0]
    
    # count the number of neighbors to get the proper sizing
    neigh_count = ti.ndarray(dtype=dtype_int, shape=(N,))
    count_neighbors_naive(pos, neigh_count, neigh_rad, dist_func, box_size)
    
    # set the neighbor arrays
    neigh_ids, neigh_offset = init_neighbor_arrays(neigh_count)

    # fill the neighbor ids
    assign_neighbor_ids_naive(pos, neigh_ids, neigh_offset, neigh_rad, dist_func, box_size)

    return neigh_ids, neigh_offset

@ti.kernel
def assign_cell_ids(
    pos: ndarray_2d_float,
    cell_ids: ndarray_1d_int,
    box_size: vec2_float,
    cell_dimensions: vec2_int):
    N = pos.shape[0]
    for i in range(N):
        frac = ti.math.mod(pos[i] / box_size, 1.0)
        cellf = ti.math.floor(frac * cell_dimensions)
        cx = ti.cast(cellf[0], dtype_int)
        cy = ti.cast(cellf[1], dtype_int)
        cell_ids[i] = cx + cy * cell_dimensions[0]
