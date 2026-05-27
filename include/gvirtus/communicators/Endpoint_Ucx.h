/*
 * Endpoint_Ucx — configuration object for the UCX communicator endpoint.
 *
 * Parsed from etc/properties_ucx.json at startup. Carries the address and port
 * that UcxCommunicator uses for ucp_listener_create (server) or ucp_ep_create
 * (client). Suite string "ucx" triggers CommunicatorFactory to dlopen the UCX
 * communicator shared library.
 *
 * Optimization phase: 1 (baseline UCX integration)
 */
#pragma once

#include <nlohmann/json.hpp>

#include "Endpoint.h"

namespace gvirtus::communicators {

class Endpoint_Ucx : public Endpoint {
   public:
    Endpoint_Ucx() = default;

    explicit Endpoint_Ucx(const std::string &endp_suite, const std::string &endp_protocol,
                          const std::string &endp_address, const std::string &endp_port);

    explicit Endpoint_Ucx(const std::string &endp_suite)
        : Endpoint_Ucx(endp_suite, "ucx", "127.0.0.1", "9999") {}

    Endpoint &suite(const std::string &suite) override;

    Endpoint &protocol(const std::string &protocol) override;

    Endpoint_Ucx &address(const std::string &address);

    Endpoint_Ucx &port(const std::string &port);

    inline const std::string &address() const { return _address; }

    inline const std::uint16_t &port() const { return _port; }

    inline const std::string to_string() const override {
        return _suite + ":" + _protocol + "://" + _address + ":" + std::to_string(_port);
    }

   private:
    std::string _address;
    std::uint16_t _port{};
};

void from_json(const nlohmann::json &j, Endpoint_Ucx &end);

}  // namespace gvirtus::communicators
