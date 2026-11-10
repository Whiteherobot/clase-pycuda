"""Ejemplo mínimo de PyCUDA: suma de dos vectores en la GPU y comparación con NumPy."""
import numpy as np
import pycuda.autoinit  # noqa: F401  (inicializa el contexto de la GPU)
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

modulo = SourceModule("""
__global__ void sumar(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
""")
sumar = modulo.get_function("sumar")

n = 1_000_000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.empty_like(a)

hilos = 256
bloques = (n + hilos - 1) // hilos
sumar(cuda.Out(c), cuda.In(a), cuda.In(b), np.int32(n), block=(hilos, 1, 1), grid=(bloques, 1))

print("GPU:", cuda.Device(0).name())
print("Resultado correcto:", np.allclose(c, a + b))
