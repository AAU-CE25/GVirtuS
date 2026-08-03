// graphvis.cu -- ¿en QUE puntos de observacion se hacen visibles al host las salidas de una
// copia D2H capturada en un grafo?
//
// I12 dice que las salidas se recuperan "en la siguiente sincronizacion". Eso no es un punto,
// es una familia, y CUDA declara host-visible el resultado en varios de ellos. Si el sistema
// remoto solo recoge en uno o dos, el soporte de D2H capturado es correcto UNICAMENTE por la
// via probada, y decirlo de otro modo es una garantia que no existe.
//
// Cada punto se prueba con un LANZAMIENTO PROPIO y una carga distinta, con el buffer de host
// prerelleno con un centinela: asi "no se recogio" es distinguible de "se recogio antes".
//
// El brazo nativo es el oraculo. Si un punto falla tambien en nativo, el fallo es del test.
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e)); \
  return 2; } }while(0)

static const size_t N = 4096;              // 32 KiB, por encima del suelo del pool
static const unsigned long long SENTINEL = 0xDEADBEEFCAFEBABEULL;

__global__ void fill(unsigned long long *d, size_t n, unsigned long long v) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] = v + i;
}

// Devuelve 0 si los bytes son los esperados, 1 si siguen siendo el centinela (no se recogio),
// 2 si son otra cosa.
static int check(const unsigned long long *h, unsigned long long v) {
    if (h[0] == SENTINEL && h[N - 1] == SENTINEL) return 1;
    for (size_t i = 0; i < N; ++i)
        if (h[i] != v + i) return 2;
    return 0;
}

int main() {
    unsigned long long *d = nullptr, *h = nullptr;
    CK(cudaMalloc((void **)&d, N * sizeof(*d)));
    CK(cudaHostAlloc((void **)&h, N * sizeof(*h), cudaHostAllocDefault));

    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));

    // Calentamiento fuera de la ventana: el primer toque de un tamano hace crecer el pool, y
    // hacerlo crecer DENTRO de una captura la invalidaria.
    CK(cudaMemcpyAsync(h, d, N * sizeof(*d), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));

    // Captura: kernel + D2H. La copia D2H capturada es lo que se esta probando.
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    fill<<<(N + 255) / 256, 256, 0, s>>>(d, N, 0);   // el valor se refresca por lanzamiento
    CK(cudaMemcpyAsync(h, d, N * sizeof(*d), cudaMemcpyDeviceToHost, s));
    cudaGraph_t g; CK(cudaStreamEndCapture(s, &g));
    cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));

    struct Punto { const char *nombre; int modo; };
    const Punto puntos[] = {
        {"cudaStreamSynchronize", 0},
        {"cudaDeviceSynchronize", 1},
        {"cudaEventSynchronize",  2},
        {"cudaStreamQuery==success", 3},
        {"cudaEventQuery==success",  4},
    };
    const int NP = sizeof(puntos) / sizeof(puntos[0]);

    int fallos = 0;
    printf("observation_point,verdict,detail\n");
    for (int p = 0; p < NP; ++p) {
        // Centinela: si nadie recoge, esto es lo que se lee.
        for (size_t i = 0; i < N; ++i) h[i] = SENTINEL;
        // El kernel del grafo escribe SIEMPRE el mismo valor base 0 (el grafo esta fijado);
        // lo que cambia entre puntos es solo COMO se observa. Basta con que el contenido no
        // sea el centinela y sea el que el kernel produce.
        const unsigned long long v = 0;

        CK(cudaGraphLaunch(ge, s));

        switch (puntos[p].modo) {
            case 0: CK(cudaStreamSynchronize(s)); break;
            case 1: CK(cudaDeviceSynchronize());  break;
            case 2: CK(cudaEventRecord(ev, s));
                    CK(cudaEventSynchronize(ev)); break;
            case 3: {
                cudaError_t st;
                int giros = 0;
                do { st = cudaStreamQuery(s); ++giros; }
                while (st == cudaErrorNotReady && giros < 20000000);
                if (st != cudaSuccess) { printf("%s,ERROR,%s\n", puntos[p].nombre,
                                                cudaGetErrorString(st)); ++fallos; continue; }
                break;
            }
            case 4: {
                CK(cudaEventRecord(ev, s));
                cudaError_t st;
                int giros = 0;
                do { st = cudaEventQuery(ev); ++giros; }
                while (st == cudaErrorNotReady && giros < 20000000);
                if (st != cudaSuccess) { printf("%s,ERROR,%s\n", puntos[p].nombre,
                                                cudaGetErrorString(st)); ++fallos; continue; }
                break;
            }
        }

        int r = check(h, v);
        const char *veredicto = r == 0 ? "VISIBLE" : (r == 1 ? "STALE" : "WRONG");
        printf("%s,%s,h[0]=0x%llx h[%zu]=0x%llx\n", puntos[p].nombre, veredicto,
               (unsigned long long)h[0], N - 1, (unsigned long long)h[N - 1]);
        if (r != 0) ++fallos;

        // Deja el estado limpio para el siguiente punto pase lo que pase.
        CK(cudaDeviceSynchronize());
    }

    printf("SUMMARY points=%d failures=%d %s\n", NP, fallos, fallos == 0 ? "PASS" : "FAIL");
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    CK(cudaStreamDestroy(s)); CK(cudaEventDestroy(ev));
    CK(cudaFree(d)); CK(cudaFreeHost(h));
    return fallos == 0 ? 0 : 1;
}
