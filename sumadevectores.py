import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

N = 1_000_000

a = numpy.random.randn(N).astype(numpy.float32)
b = numpy.random.randn(N).astype(numpy.float32)

dev_a = cuda.mem_alloc(a.nbytes)
dev_b = cuda.mem_alloc(b.nbytes)
dev_c = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(dev_a, a)
cuda.memcpy_htod(dev_b, b)

mod = SourceModule("""
__global__ void add(const float *a, const float *b, float *c, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
""")

add = mod.get_function("add")

suma_numpy = a + b

for hilos in (128, 256, 512):
    bloques = (N + hilos - 1) // hilos

    add(dev_a, dev_b, dev_c, numpy.int32(N),
        block=(hilos, 1, 1), grid=(bloques, 1))

    c = numpy.empty_like(a)
    cuda.memcpy_dtoh(c, dev_c)

    correcto = numpy.allclose(c, suma_numpy)
    print("Hilos por bloque: %3d | Bloques: %5d | Hilos totales: %7d | Correcto: %s"
          % (hilos, bloques, hilos * bloques, correcto))

    cuda.memset_d32(dev_c, 0, N)

dev_a.free()
dev_b.free()
dev_c.free()
