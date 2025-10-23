import taichi as ti
from .config import dtype_int, ndarray_1d_int

@ti.kernel
def sum1d_i32(
    a: ndarray_1d_int,
    total: ti.template()):
    total[None] = ti.cast(0, total.dtype)
    for i in range(a.shape[0]):
        total[None] += a[i]

@ti.kernel
def exclusive_scan_i32(
    a: ndarray_1d_int,
    out: ndarray_1d_int,
    tmp: ndarray_1d_int):
    n = a.shape[0]
    for i in range(n):
        out[i] = a[i]
    offset = 1
    while offset < n:
        for i in range(n):
            val = out[i]
            if i >= offset:
                val += out[i - offset]
            tmp[i] = val
        for i in range(n):
            out[i] = tmp[i]
        offset <<= 1
    if n > 0:
        prev = ti.cast(0, dtype_int)
        for i in range(n):
            cur = out[i]
            out[i] = prev
            prev = cur
        out[n] = prev