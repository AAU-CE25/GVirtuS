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

    virtual inline const std::string to_string() const override {
        return _suite + ":" + _protocol + "://" + _address + ":" + std::to_string(_port);
    }

   private:
    std::string _address;
    std::uint16_t _port;
};

void from_json(const nlohmann::json &j, Endpoint_Ucx &end);

}  // namespace gvirtus::communicators
