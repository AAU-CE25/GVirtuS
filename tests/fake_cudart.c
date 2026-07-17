/*
 * Minimal fake libcudart, used ONLY by test_buffer_iov's GpuRef test cases
 * so the real dlsym'd DeviceMemory.cpp code path (IsDevicePointer,
 * DeviceMemcpyD2H, AllocDeviceMemory, FreeDeviceMemory) can be exercised in
 * a sandbox with no GPU/CUDA toolkit installed, instead of only asserting
 * "the probe is disabled". Any pointer obtained via this fake cudaMalloc is
 * registered; cudaPointerGetAttributes reports Device (type=2) for
 * registered pointers and Unregistered (type=0) for anything else, exactly
 * like the real driver would distinguish a cudaMalloc'd pointer from a
 * plain malloc'd one. cudaMemcpy just does a real memcpy — in this fake
 * environment "device" memory is actually ordinary process memory, so the
 * bytes really are there to copy, which is exactly what we want to verify
 * end-to-end (that DeviceMemcpyD2H's bounce path moves the right bytes).
 *
 * Build as libcudart.so.12 (the first dlopen candidate DeviceMemory.cpp
 * tries) and put its directory on LD_LIBRARY_PATH when running the test:
 *   gcc -shared -fPIC -o libcudart.so.12 tests/fake_cudart.c
 */
#include <stdlib.h>
#include <string.h>

#define MAX_REGISTRY 1024
static void *g_registry[MAX_REGISTRY];
static int g_registry_count = 0;

static int is_registered(const void *p) {
    for (int i = 0; i < g_registry_count; ++i)
        if (g_registry[i] == p) return 1;
    return 0;
}

int cudaMalloc(void **p, size_t n) {
    if (p == NULL) return 1;
    void *mem = malloc(n > 0 ? n : 1);
    if (mem == NULL) return 2;
    if (g_registry_count < MAX_REGISTRY) g_registry[g_registry_count++] = mem;
    *p = mem;
    return 0;
}

int cudaFree(void *p) {
    if (p == NULL) return 0;
    for (int i = 0; i < g_registry_count; ++i) {
        if (g_registry[i] == p) {
            g_registry[i] = g_registry[--g_registry_count];
            break;
        }
    }
    free(p);
    return 0;
}

/* Mirrors DeviceMemory.cpp's cudaPointerAttributes_layout exactly:
 *   { int type; int device; void *devicePointer; void *hostPointer; }
 * type: 0=Unregistered, 1=Host, 2=Device, 3=Managed.
 */
int cudaPointerGetAttributes(void *attrs_out, const void *p) {
    int *type = (int *)attrs_out;
    /* zero the whole struct first (4 x 8 bytes worst case with padding) */
    memset(attrs_out, 0, sizeof(int) * 2 + sizeof(void *) * 2);
    *type = is_registered(p) ? 2 /*Device*/ : 0 /*Unregistered*/;
    return 0;
}

/* kind is ignored — in this fake environment every pointer is real process
 * memory, so a plain memcpy is a faithful stand-in for every cudaMemcpy
 * direction (H2D/D2H/D2D). */
int cudaMemcpy(void *dst, const void *src, size_t n, int kind) {
    (void)kind;
    if (n == 0) return 0;
    if (dst == NULL || src == NULL) return 1;
    memcpy(dst, src, n);
    return 0;
}
