import taichi as ti
import platform

# Select GPU backend
arch = ti.metal if platform.system() == "Darwin" else ti.cuda
ti.init(arch=arch)
print("taichi:", ti.__version__)

# Define common types
dtype_float = ti.f32
dtype_int = ti.int32

vec2_int = ti.types.vector(2, dtype_int)
vec2_float = ti.types.vector(2, dtype_float)

ndarray_1d_int = ti.types.ndarray(dtype=dtype_int, ndim=1)
ndarray_1d_float = ti.types.ndarray(dtype=dtype_float, ndim=1)

ndarray_2d_int = ti.types.ndarray(dtype=vec2_int, ndim=1)
ndarray_2d_float = ti.types.ndarray(dtype=vec2_float, ndim=1)