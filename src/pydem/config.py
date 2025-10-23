import taichi as ti
import numpy as np
import platform

# Select backend
# ti.init(arch=ti.metal)
# ti.init(arch=ti.cuda)
ti.init(arch=ti.cpu)
print("taichi:", ti.__version__)

# Define common types
dtype_int = ti.int32
dtype_int_np = np.int32

prec = 64
if prec == 32:
    dtype_float = ti.float32
    dtype_float_np = np.float32
elif prec == 64:
    dtype_float = ti.float64
    dtype_float_np = np.float64

vec2_int = ti.types.vector(2, dtype_int)
vec2_float = ti.types.vector(2, dtype_float)

ndarray_1d_int = ti.types.ndarray(dtype=dtype_int, ndim=1)
ndarray_1d_float = ti.types.ndarray(dtype=dtype_float, ndim=1)

ndarray_2d_int = ti.types.ndarray(dtype=vec2_int, ndim=1)
ndarray_2d_float = ti.types.ndarray(dtype=vec2_float, ndim=1)