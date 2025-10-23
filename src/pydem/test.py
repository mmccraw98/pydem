import taichi as ti
import numpy as np
from .config import vec2_float, dtype_float, vec2_int
from .neighbors import build_neighbors_naive
from .box import dist_pbc

if __name__ == "__main__":
    N = 100
    phi = 0.5

    radii_np = np.ones(N, dtype=np.float32) * 0.5
    radii_np[N // 2:] *= 1.4
    radii = ti.ndarray(dtype=dtype_float, shape=(N,))
    radii.from_numpy(radii_np)

    box_size_np = np.ones(2, dtype=np.float32) * np.sqrt(np.sum(np.pi * radii_np ** 2) / phi)
    box_size = vec2_float(box_size_np[0], box_size_np[1])

    pos_np = np.random.uniform(low=0, high=1, size=(N, 2)).astype(np.float32)
    pos_np *= box_size_np
    pos = ti.ndarray(dtype=vec2_float, shape=(N,))
    pos.from_numpy(pos_np)

    neigh_ids, neigh_offset = build_neighbors_naive(pos, 1.0, dist_pbc, box_size)

