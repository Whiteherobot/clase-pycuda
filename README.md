# Clase PyCUDA

Computación Paralela · Michael Lata

Ejemplos de clase con PyCUDA y el ejercicio de suma de vectores grandes en la GPU, desde Python.

## Ejercicio: suma de vectores

`sumadevectores.py` suma dos vectores de 1 000 000 de elementos (`float32`, creados con NumPy) en la GPU:

1. Pasa los vectores al kernel con `drv.In(a)` y `drv.In(b)`, y el resultado con `drv.Out(dest)`. PyCUDA reserva la memoria en la GPU, copia las entradas antes de lanzar el kernel, trae `dest` de vuelta al terminar y libera la memoria.
2. Compila el kernel `add` con `SourceModule`. Cada hilo calcula su índice global `threadIdx.x + blockIdx.x * blockDim.x` y solo escribe si `tid < n`, para no salirse del vector.
3. Lo lanza con 128, 256 y 512 hilos por bloque. Los bloques se calculan como `(N + hilos - 1) // hilos`.
4. Compara el resultado con `a + b` de NumPy usando `numpy.allclose`.

Resultado en una RTX 4060:

| Hilos por bloque | Bloques | Hilos totales | Correcto |
|---:|---:|---:|:---:|
| 128 | 7813 | 1 000 064 | ✔ |
| 256 | 3907 | 1 000 192 | ✔ |
| 512 | 1954 | 1 000 448 | ✔ |

Siempre se lanzan algunos hilos de más que elementos; el `if (tid < n)` evita que esos hilos sobrantes escriban fuera del vector.

## Ejecutar

Requiere una GPU NVIDIA con el toolkit de CUDA (`nvcc`) instalado.

```bash
pip install -r requirements.txt
python sumadevectores.py
```

## Archivos

| Archivo | Contenido |
|---|---|
| `sumadevectores.py` | ejercicio de suma de vectores con 128, 256 y 512 hilos por bloque |
| `ejemplo1.py` | duplica una matriz de 5×5 con un bloque de 5×5 hilos |
| `hola_gpu.py` | suma mínima de vectores usando `cuda.In` / `cuda.Out` |
| `info_gpu.py` | lista las GPU detectadas, su compute capability y su memoria |

## Licencia

MIT. Ver [`LICENSE`](LICENSE).
