#pragma once

#include <gvirtus/common/LD_Lib.h>
#include <gvirtus/backend/Handler.h>
#include <gvirtus/communicators/FramedStream.h>
#include <ucp/api/ucp.h>
#include <vector>
#include <string>

namespace gvirtus::backend {

class UcxProcess {
   public:
    UcxProcess(uint16_t port, std::vector<std::string> plugins);
    ~UcxProcess();
    void Start();

   private:
    void ServeConnection(ucp_worker_h worker, ucp_ep_h ep);
    std::vector<uint8_t> SerializeResult(std::shared_ptr<communicators::Result> result);

    uint16_t port_;
    std::vector<std::string> plugin_names_;
    std::vector<std::shared_ptr<common::LD_Lib<Handler>>> handlers_;

    ucp_context_h ucp_context_{nullptr};
    ucp_worker_h  ucp_worker_{nullptr};
    ucp_listener_h listener_{nullptr};

    log4cplus::Logger logger_;
};

}  // namespace gvirtus::backend
