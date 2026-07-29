/*
 * gVirtuS -- cache local de consultas del Driver API. Ver CudaDr_querycache.h.
 */

#include "CudaDr_querycache.h"

#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>

namespace gvirtus_drqcache {
namespace {

std::mutex &mu() {
    static std::mutex m;
    return m;
}

/* (atributo, dispositivo) -> valor. Inmutable, nunca se invalida. */
std::map<std::pair<int, int>, int> &attrs() {
    static std::map<std::pair<int, int>, int> m;
    return m;
}

bool g_ctx_valid = false;
CUcontext g_ctx = nullptr;
bool g_dev_valid = false;
CUdevice g_dev = 0;

}  // namespace

bool enabled() {
    static const bool on = [] {
        const char *v = std::getenv("GVIRTUS_LOCAL_QUERY_CACHE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

bool attr_get(int attrib, int dev, int *out) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = attrs().find({attrib, dev});
    if (it == attrs().end()) return false;
    if (out != nullptr) *out = it->second;
    return true;
}

void attr_put(int attrib, int dev, int value) {
    std::lock_guard<std::mutex> lk(mu());
    attrs()[{attrib, dev}] = value;
}

bool ctx_get_current(CUcontext *out) {
    std::lock_guard<std::mutex> lk(mu());
    if (!g_ctx_valid) return false;
    if (out != nullptr) *out = g_ctx;
    return true;
}

void ctx_put_current(CUcontext v) {
    std::lock_guard<std::mutex> lk(mu());
    g_ctx = v;
    g_ctx_valid = true;
}

bool ctx_get_device(CUdevice *out) {
    std::lock_guard<std::mutex> lk(mu());
    if (!g_dev_valid) return false;
    if (out != nullptr) *out = g_dev;
    return true;
}

void ctx_put_device(CUdevice v) {
    std::lock_guard<std::mutex> lk(mu());
    g_dev = v;
    g_dev_valid = true;
}

/* Falla cerrado: al invalidar se olvida todo lo que depende del contexto, de modo que la
 * siguiente consulta vuelve a preguntar al backend. Los atributos NO se tocan: son del
 * dispositivo, no del contexto, y no cambian. */
void ctx_invalidate() {
    std::lock_guard<std::mutex> lk(mu());
    g_ctx_valid = false;
    g_dev_valid = false;
}

}  // namespace gvirtus_drqcache
