/*
 * gVirtuS -- envio diferido de fatbins. Ver CudaRt_lazyfatbin.h para el porque.
 */

#include "CudaRt_lazyfatbin.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "CudaRt.h"
#include "CudaUtil.h"
#include <gvirtus/common/FatbinHash.h>

using gvirtus::communicators::Buffer;

namespace gvirtus_lazyfat {
namespace {

struct FuncReg {
    const void *hostFun;
    std::string deviceFun;
    std::string deviceName;
    int thread_limit;
    uint3 *tid;
    uint3 *bid;
    dim3 *bDim;
    dim3 *gDim;
    int *wSize;
};

struct VarReg {
    char *hostVar;
    std::string deviceAddress;
    std::string deviceName;
    int ext, size, constant, global;
};

struct Pending {
    void *bin = nullptr;
    bool end_called = false;
    bool shipped = false;
    std::vector<FuncReg> funcs;
    std::vector<VarReg> vars;
};

std::mutex &mu() {
    static std::mutex m;
    return m;
}
std::map<void **, Pending> &pend() {
    static std::map<void **, Pending> m;
    return m;
}
std::map<const void *, void **> &owner() {
    static std::map<const void *, void **> m;
    return m;
}

std::atomic<long> g_total{0}, g_shipped{0};
std::atomic<long> g_dedup_hits{0}, g_dedup_miss{0};

/* Envia UNA entrada. El llamante debe tener mu() tomado. El orden replica el de CUDA:
 * fatbin, funciones, variables y End al final. Ver la nota junto al bloque de End. */
void ship_locked(void **handle, Pending &p) {
    if (p.shipped) return;
    p.shipped = true;
    g_shipped.fetch_add(1);

    __fatBinC_Wrapper_t *bin = static_cast<__fatBinC_Wrapper_t *>(p.bin);

    if (dedup_enabled() && dedup_probe(bin)) {
        /* El backend ya tenia este contenido y ha ligado nuestro handler: no hace falta
         * enviar el blob. Las funciones y variables SI se registran despues, porque sus
         * punteros hostFun son propios de este cliente. */
    } else {
        Buffer *ib = new Buffer();
        ib->AddString(CudaUtil::MarshalHostPointer(reinterpret_cast<void **>(bin)));
        ib = CudaUtil::MarshalFatCudaBinary(bin, ib);
        CudaRtFrontend::Prepare();
        CudaRtFrontend::Execute("cudaRegisterFatBinary", ib);
        if (dedup_enabled()) dedup_record(bin);
    }

    for (auto &f : p.funcs) {
        CudaRtFrontend::Prepare();
        CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(handle));
        CudaRtFrontend::AddVariableForArguments((gvirtus::common::pointer_t)f.hostFun);
        CudaRtFrontend::AddStringForArguments(f.deviceFun.c_str());
        CudaRtFrontend::AddStringForArguments(f.deviceName.c_str());
        CudaRtFrontend::AddVariableForArguments(f.thread_limit);
        CudaRtFrontend::AddHostPointerForArguments(f.tid);
        CudaRtFrontend::AddHostPointerForArguments(f.bid);
        CudaRtFrontend::AddHostPointerForArguments(f.bDim);
        CudaRtFrontend::AddHostPointerForArguments(f.gDim);
        CudaRtFrontend::AddHostPointerForArguments(f.wSize);
        CudaRtFrontend::Execute("cudaRegisterFunction");
    }

    for (auto &v : p.vars) {
        CudaRtFrontend::Prepare();
        CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(handle));
        CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(v.hostVar));
        CudaRtFrontend::AddStringForArguments(v.deviceAddress.c_str());
        CudaRtFrontend::AddStringForArguments(v.deviceName.c_str());
        CudaRtFrontend::AddVariableForArguments(v.ext);
        CudaRtFrontend::AddVariableForArguments(v.size);
        CudaRtFrontend::AddVariableForArguments(v.constant);
        CudaRtFrontend::AddVariableForArguments(v.global);
        CudaRtFrontend::Execute("cudaRegisterVar");
    }

    // End va AL FINAL, no justo tras el fatbin. CUDA llama en el orden
    // RegisterFatBinary -> RegisterFunction/Var -> RegisterFatBinaryEnd, y el backend
    // finaliza el modulo en End: las funciones registradas despues se pierden. Enviarlo
    // en segundo lugar era lo que producia cudaErrorInvalidDeviceFunction en el ETL.
    if (p.end_called) {
        Buffer *eb = new Buffer();
        eb->AddString(CudaUtil::MarshalHostPointer(reinterpret_cast<void **>(bin)));
        eb = CudaUtil::MarshalFatCudaBinary(bin, eb);
        CudaRtFrontend::Prepare();
        CudaRtFrontend::Execute("cudaRegisterFatBinaryEnd", eb);
    }
}

}  // namespace

namespace {
gvirtus::common::FatbinKey key_of(void *binv) {
    const __fatBinC_Wrapper_t *bin = static_cast<const __fatBinC_Wrapper_t *>(binv);
    const struct fatBinaryHeader *hdr =
        reinterpret_cast<const struct fatBinaryHeader *>(bin->data);
    return gvirtus::common::fatbin_key(bin->data, hdr->headerSize + hdr->fatSize);
}
}  // namespace

bool dedup_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("GVIRTUS_FATBIN_DEDUP");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

bool dedup_probe(void *binv) {
    const gvirtus::common::FatbinKey k = key_of(binv);
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(
        CudaUtil::MarshalHostPointer(reinterpret_cast<void **>(binv)));
    CudaRtFrontend::AddVariableForArguments(k.len);
    CudaRtFrontend::AddVariableForArguments(k.h1);
    CudaRtFrontend::AddVariableForArguments(k.h2);
    CudaRtFrontend::Execute("cudaRegisterFatBinaryProbe");
    /* Un backend sin esta rutina responde con error: se trata como fallo de cache y se
     * envia por el camino de siempre. Tolerante a desfase de versiones entre extremos. */
    if (!CudaRtFrontend::Success()) { g_dedup_miss.fetch_add(1); return false; }
    int hit = 0;
    try { hit = CudaRtFrontend::GetOutputVariable<int>(); } catch (...) { hit = 0; }
    if (hit == 1) { g_dedup_hits.fetch_add(1); return true; }
    g_dedup_miss.fetch_add(1);
    return false;
}

void dedup_record(void *binv) {
    const gvirtus::common::FatbinKey k = key_of(binv);
    /* Indexar DESPUES de registrar: si el registro fallara, no queremos que otro cliente
     * reciba un acierto sobre un modulo que no existe. */
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(
        CudaUtil::MarshalHostPointer(reinterpret_cast<void **>(binv)));
    CudaRtFrontend::AddVariableForArguments(k.len);
    CudaRtFrontend::AddVariableForArguments(k.h1);
    CudaRtFrontend::AddVariableForArguments(k.h2);
    CudaRtFrontend::Execute("cudaRegisterFatBinaryBind");
}

bool enabled() {
    static const bool on = [] {
        const char *v = std::getenv("GVIRTUS_LAZY_FATBIN");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

void note_fatbin(void **handle, void *bin) {
    std::lock_guard<std::mutex> lk(mu());
    Pending &p = pend()[handle];
    p.bin = bin;
    g_total.fetch_add(1);
}

void note_fatbin_end(void **handle, void *bin) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = pend().find(handle);
    if (it == pend().end()) return;
    it->second.end_called = true;
    if (it->second.bin == nullptr) it->second.bin = bin;
}

bool note_function(void **handle, const void *hostFun, char *deviceFun, const char *deviceName,
                   int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = pend().find(handle);
    if (it == pend().end()) return false;  // fatbin no diferido: que lo envie el llamante
    it->second.funcs.push_back(FuncReg{hostFun, deviceFun ? deviceFun : "",
                                       deviceName ? deviceName : "", thread_limit, tid, bid, bDim,
                                       gDim, wSize});
    owner()[hostFun] = handle;
    return true;
}

bool note_var(void **handle, char *hostVar, char *deviceAddress, const char *deviceName, int ext,
              int size, int constant, int global) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = pend().find(handle);
    if (it == pend().end()) return false;
    it->second.vars.push_back(VarReg{hostVar, deviceAddress ? deviceAddress : "",
                                     deviceName ? deviceName : "", ext, size, constant, global});
    return true;
}

void ensure_for_hostfun(const void *hostFun) {
    std::lock_guard<std::mutex> lk(mu());
    auto o = owner().find(hostFun);
    if (o == owner().end()) {
        // No sabemos de quien es esta funcion. Puede venir de una ruta de registro que no
        // interceptamos, o de un lanzamiento que no pasa por nuestros ganchos. Enviar todo
        // lo pendiente es la unica respuesta segura: renuncia a la ganancia para ESTE
        // proceso, pero nunca lanza contra un backend que no conoce el modulo.
        //
        // Sin esta red, el fallo es cudaErrorInvalidDeviceFunction en medio del workload,
        // que es justo como se manifesto la primera version de esto.
        for (auto &kv : pend()) {
            if (!kv.second.shipped) ship_locked(kv.first, kv.second);
        }
        return;
    }
    auto it = pend().find(o->second);
    if (it == pend().end() || it->second.shipped) return;
    ship_locked(o->second, it->second);
}

void flush_all() {
    std::lock_guard<std::mutex> lk(mu());
    for (auto &kv : pend()) {
        if (!kv.second.shipped) ship_locked(kv.first, kv.second);
    }
}

void report_stats() {
    const char *v = std::getenv("GVIRTUS_LAZY_FATBIN_STATS");
    if (v == nullptr || v[0] != '1') return;
    std::fprintf(stderr,
                 "[GVS LAZY FATBIN] sent %ld of %ld registered | dedup: %ld hits, "
                 "%ld misses\n",
                 g_shipped.load(), g_total.load(), g_dedup_hits.load(), g_dedup_miss.load());
    std::fflush(stderr);
}

}  // namespace gvirtus_lazyfat
