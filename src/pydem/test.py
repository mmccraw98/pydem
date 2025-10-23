import taichi as ti
import numpy as np
from .config import vec2_float, dtype_float, vec2_int
from .neighbors import build_neighbors_naive
from .box import dist_pbc
from .potential import soft_disk
from .core import compute_forces

if __name__ == "__main__":
    N = 1000
    phi = 0.5
    e_c = 1.0

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

    force = ti.ndarray(dtype=vec2_float, shape=(N,))
    force.fill(0)
    pe = ti.ndarray(dtype=dtype_float, shape=(N,))
    pe.fill(0)

    compute_forces(pos, force, pe, radii, neigh_ids, neigh_offset, dist_pbc, box_size, soft_disk, e_c)
    
    print(np.sum(force.to_numpy(), axis=0))

