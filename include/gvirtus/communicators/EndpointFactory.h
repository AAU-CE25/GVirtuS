#pragma once

#include <gvirtus/common/JSON.h>
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>

#include "Endpoint.h"
#include "Endpoint_Hybrid.h"
#include "Endpoint_Rdma.h"
#include "Endpoint_Tcp.h"
#include "Endpoint_Ucx.h"

// #define DEBUG

namespace gvirtus::communicators {

class EndpointFactory {
   public:
    static std::shared_ptr<Endpoint> get_endpoint(const fs::path &json_path) {
#ifdef DEBUG
        std::cout << "EndpointFactory::get_endpoint() called" << std::endl;
#endif

        std::shared_ptr<Endpoint> ptr;
        std::ifstream ifs(json_path);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open configuration file: " + json_path.string());
        }

        nlohmann::json j;
        ifs >> j;

        log4cplus::Logger logger =
            log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS.EndpointFactory"));
        LOG4CPLUS_INFO(logger, "Parsing endpoint config from: " + json_path.string());
        LOG4CPLUS_DEBUG(logger, "Full JSON dump:\n" + j.dump(4));

        if (!j.contains("communicator") || !j["communicator"].is_array() ||
            j["communicator"].empty()) {
            throw std::runtime_error("Invalid or missing 'communicator' array in configuration.");
        }

        const auto &endpoint_obj = j["communicator"][0]["endpoint"];
        if (!endpoint_obj.contains("suite") || endpoint_obj["suite"].is_null()) {
            throw std::runtime_error("Missing or null 'suite' in endpoint configuration.");
        }

        std::string suite = endpoint_obj["suite"].get<std::string>();
        LOG4CPLUS_INFO(logger, "Parsed endpoint suite: " + suite);

        if (suite == "tcp/ip") {
            LOG4CPLUS_INFO(logger, "Initializing TCP/IP Endpoint");
            auto end = common::JSON<Endpoint_Tcp>(json_path).parser();
            ptr = std::make_shared<Endpoint_Tcp>(end);
        }
        // infiniband
        else if ("infiniband-rdma" == j["communicator"][ind_endpoint]["endpoint"].at("suite")) {
#ifdef DEBUG
            std::cout << "EndpointFactory::get_endpoint() found infiniband endpoint" << std::endl;
#endif
            auto end = common::JSON<Endpoint_Rdma>(json_path).parser();
            ptr = std::make_shared<Endpoint_Rdma>(end);
        } else if ("roce-rdma" == j["communicator"][ind_endpoint]["endpoint"].at("suite")) {
#ifdef DEBUG
            std::cout << "EndpointFactory::get_endpoint() found rdma-roce endpoint (reusing "
                         "Endpoint_Rdma)"
                      << std::endl;
#endif
            auto end = common::JSON<Endpoint_Rdma>(json_path).parser();
            ptr = std::make_shared<Endpoint_Rdma>(end);
        } else if ("hybrid" == j["communicator"][ind_endpoint]["endpoint"].at("suite")) {
#ifdef DEBUG
            std::cout << "EndpointFactory::get_endpoint() found hybrid endpoint" << std::endl;
#endif
            auto end = common::JSON<Endpoint_Hybrid>(json_path).parser();
            ptr = std::make_shared<Endpoint_Hybrid>(end);
        } else if ("ucx" == j["communicator"][ind_endpoint]["endpoint"].at("suite")) {
            LOG4CPLUS_INFO(logger, "Initializing UCX Endpoint");
            auto end = common::JSON<Endpoint_Ucx>(json_path).parser();
            ptr = std::make_shared<Endpoint_Ucx>(end);
        } else {
            throw std::runtime_error(
                "EndpointFactory::get_endpoint(): Your suite is not compatible!");
        }

        ind_endpoint++;

        j.clear();
        ifs.close();

#ifdef DEBUG
        std::cout << "EndpointFactoru::get_endpoint(): end is: " << ptr->to_string() << std::endl;
        std::cout << "EndpointFactory::get_endpoint() ended" << std::endl;
#endif

        return ptr;
    }

    static int index() { return ind_endpoint; }

   private:
    static int ind_endpoint;
};
}  // namespace gvirtus::communicators