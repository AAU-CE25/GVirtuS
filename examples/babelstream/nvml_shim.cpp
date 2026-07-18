// Cosmetic no-op NVML shim for running BabelStream through GVirtuS.
//
// BabelStream calls a few NVML functions (nvmlInit, nvmlDeviceGetHandleByPciBusId,
// nvmlDeviceGetClockInfo) ONLY to print a theoretical peak-bandwidth line at startup.
// GVirtuS' nvml plugin does not implement GetHandleByPciBusId_v2 / GetClockInfo, and
// NVML is not part of the measured CUDA path. Linking this shim instead of -lnvidia-ml
// satisfies the symbols locally so the (purely cosmetic) startup print succeeds without
// a round-trip. The reported "PEAK" line is therefore not meaningful under GVirtuS.
extern "C" {
typedef int nvmlReturn_t;
nvmlReturn_t nvmlInit_v2(void) { return 0; }
nvmlReturn_t nvmlShutdown(void) { return 0; }
nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *busid, void **dev) {
    if (dev) *dev = (void *)1;
    return 0;
}
nvmlReturn_t nvmlDeviceGetClockInfo(void *dev, int type, unsigned int *clockMHz) {
    if (clockMHz) *clockMHz = 9001;
    return 0;
}
const char *nvmlErrorString(nvmlReturn_t r) { return "nvml-shim"; }
}
