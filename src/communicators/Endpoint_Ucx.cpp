#include "gvirtus/communicators/Endpoint_Ucx.h"

#include <regex>

#include "gvirtus/communicators/EndpointFactory.h"

using gvirtus::communicators::Endpoint;
using gvirtus::communicators::Endpoint_Ucx;
using gvirtus::communicators::EndpointFactory;

Endpoint_Ucx::Endpoint_Ucx(const std::string &endp_suite, const std::string &endp_protocol,
                           const std::string &endp_address, const std::string &endp_port) {
    suite(endp_suite);
    protocol(endp_protocol);
    address(endp_address);
    port(endp_port);
}

Endpoint &Endpoint_Ucx::suite(const std::string &suite) {
    _suite = suite;
    return *this;
}

Endpoint &Endpoint_Ucx::protocol(const std::string &protocol) {
    _protocol = protocol;
    return *this;
}

Endpoint_Ucx &Endpoint_Ucx::address(const std::string &address) {
    std::regex pattern{
        R"(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)"};
    std::smatch matches;
    std::regex_search(address, matches, pattern);
    if (address == matches[0]) _address = address;
    return *this;
}

Endpoint_Ucx &Endpoint_Ucx::port(const std::string &port) {
    std::regex pattern{
        R"((6553[0-5]|655[0-2][0-9]\d|65[0-4](\d){2}|6[0-4](\d){3}|[1-5](\d){4}|[1-9](\d){0,3}))"};
    std::smatch matches;
    std::regex_search(port, matches, pattern);
    if (port == matches[0]) _port = (uint16_t)std::stoi(port);
    return *this;
}

void gvirtus::communicators::from_json(const nlohmann::json &j, Endpoint_Ucx &end) {
    auto el = j["communicator"][EndpointFactory::index()]["endpoint"];
    end.suite(el.at("suite"));
    end.protocol(el.at("protocol"));
    end.address(el.at("server_address"));
    end.port(el.at("port"));
}
