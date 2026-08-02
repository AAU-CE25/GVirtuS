//
// Created by Mariano Aponte on 18/12/23.
//

#include "gvirtus/communicators/Endpoint_Rdma.h"

#include <regex>

#include "gvirtus/communicators/EndpointFactory.h"
#include "gvirtus/communicators/Endpoint_Tcp.h"

gvirtus::communicators::Endpoint_Rdma::Endpoint_Rdma(const std::string &endp_suite,
                                                     const std::string &endp_protocol,
                                                     const std::string &endp_address,
                                                     const std::string &endp_port) {
    suite(endp_suite);
    protocol(endp_protocol);
    address(endp_address);
    port(endp_port);
}

gvirtus::communicators::Endpoint &gvirtus::communicators::Endpoint_Rdma::suite(
    const std::string &suite) {
    std::regex pattern{R"([[:alpha:]]*-[[:alpha:]]*)"};

    std::smatch matches;

    std::regex_search(suite, matches, pattern);

    if (suite == matches[0]) _suite = suite;

    return *this;
}

gvirtus::communicators::Endpoint &gvirtus::communicators::Endpoint_Rdma::protocol(
    const std::string &protocol) {
    std::regex pattern{R"([[:alpha:]]*)"};

    std::smatch matches;

    std::regex_search(protocol, matches, pattern);

    if (protocol == matches[0]) _protocol = protocol;

    return *this;
}

gvirtus::communicators::Endpoint_Rdma &gvirtus::communicators::Endpoint_Rdma::address(
    const std::string &address) {
    std::regex pattern{
        R"(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)"};
    std::smatch matches;

    std::regex_search(address, matches, pattern);

    if (address == matches[0]) _address = address;

    return *this;
}

gvirtus::communicators::Endpoint_Rdma &gvirtus::communicators::Endpoint_Rdma::port(
    const std::string &port) {
    std::regex pattern{
        R"((6553[0-5]|655[0-2][0-9]\d|65[0-4](\d){2}|6[0-4](\d){3}|[1-5](\d){4}|[1-9](\d){0,3}))"};

    std::smatch matches;

    std::regex_search(port, matches, pattern);

    if (port == matches[0]) _port = (uint16_t)std::stoi(port);

    return *this;
}

void gvirtus::communicators::from_json(const nlohmann::json &j, Endpoint_Rdma &end) {
    // El indice se reduce AQUI tambien. get_endpoint() aplica el modulo sobre el contador
    // estatico, pero entre esa reduccion y esta lectura otros hilos lo han incrementado, asi
    // que en crudo se sale del array (heap-buffer-overflow medido con >=24 hilos). .at()
    // convierte cualquier indice malo que quede en excepcion, no en lectura fuera de rango.
    const auto &arr = j.at("communicator");
    const std::size_t idx =
        arr.empty() ? 0u : (static_cast<std::size_t>(EndpointFactory::index()) % arr.size());
    auto el = arr.at(idx).at("endpoint");

    end.suite(el.at("suite"));
    end.protocol(el.at("protocol"));
    end.address(el.at("server_address"));
    end.port(el.at("port"));
}