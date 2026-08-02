#include <atomic>
#include "gvirtus/communicators/EndpointFactory.h"

std::atomic<int> gvirtus::communicators::EndpointFactory::ind_endpoint{0};
