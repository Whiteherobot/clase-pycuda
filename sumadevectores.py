import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule

N = 1_000_000

a = numpy.random.randn(N).astype(numpy.float32)
b = numpy.random.randn(N).astype(numpy.float32)

mod = SourceModule("""
__global__ void add(float *dest, float *a, float *b, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        dest[tid] = a[tid] + b[tid];
    }
}
""")

add = mod.get_function("add")

suma_numpy = a + b

for hilos in (128, 256, 512):
    bloques = (N + hilos - 1) // hilos

    dest = numpy.zeros_like(a)
    add(
        drv.Out(dest), drv.In(a), drv.In(b), numpy.int32(N),
        block=(hilos, 1, 1), grid=(bloques, 1))

    correcto = numpy.allclose(dest, suma_numpy)
    print("Hilos por bloque: %3d | Bloques: %5d | Hilos totales: %7d | Correcto: %s"
          % (hilos, bloques, hilos * bloques, correcto))
