/**
 * @mainpage gVirtuS - A GPGPU transparent virtualization component
 *
 * @section Introduction
 * gVirtuS tries to fill the gap between in-house hosted computing clusters,
 * equipped with GPGPUs devices, and pay-for-use high performance virtual
 * clusters deployed  via public or private computing clouds. gVirtuS allows an
 * instanced virtual machine to access GPGPUs in a transparent way, with an
 * overhead  slightly greater than a real machine/GPGPU setup. gVirtuS is
 * hypervisor independent, and, even though it currently virtualizes nVIDIA CUDA
 * based GPUs, it is not limited to a specific brand technology. The performance
 * of the components of gVirtuS is assessed through a suite of tests in
 * different deployment scenarios, such as providing GPGPU power to cloud
 * computing based HPC clusters and sharing remotely hosted GPGPUs among HPC
 * nodes.
 *
 * Written By: Carlo Palmieri <carlo.palmieri@uniparthenope.it>,
 *             Department of Applied Science
 *             Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 *             Raffaele Montella <raffaele.montella@uniparthenope.it>,
 *             Department of Science and Technologies
 *             Antonio Mentone <antonio.mentone@uniparthenope.it>,
 *             Department of Science and Technologies
 * Edited By: Mariano Aponte <aponte2001@gmail.com>,
 *            Department of Science and Technologies, University of Naples Parthenope
 *            Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            Department of Computer Science, University College Dublin
 */

#include <log4cplus/consoleappender.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

#include "gvirtus/backend/Backend.h"
#include "gvirtus/backend/Property.h"
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using namespace log4cplus;

Logger logger;

std::string getEnvVar(const std::string& name) {
    const char* val = std::getenv(name.c_str());
    return val ? val : "";
}

std::string trim_copy(const std::string& s) {
    std::size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    std::size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool has_component_token(const std::string& scope, const char* token) {
    std::stringstream ss(scope);
    std::string item;
    const std::string wanted = to_lower_copy(token);
    while (std::getline(ss, item, ',')) {
        if (to_lower_copy(trim_copy(item)) == wanted) return true;
    }
    return false;
}

void apply_debug_component_scope(LogLevel root_level) {
    // Scope filtering only applies when DEBUG/TRACE is enabled globally.
    if (root_level > DEBUG_LOG_LEVEL) return;

    const std::string scope = getEnvVar("GVIRTUS_DEBUG_SCOPE");
    if (scope.empty()) return;
    if (has_component_token(scope, "all")) return;

    const bool backend = has_component_token(scope, "backend");
    const bool frontend = has_component_token(scope, "frontend");
    const bool communicator = has_component_token(scope, "communicator") ||
                              has_component_token(scope, "comm") ||
                              has_component_token(scope, "ucx") ||
                              has_component_token(scope, "tcp");

    if (!backend && !frontend && !communicator) {
        std::cerr << "[GVIRTUS WARNING] GVIRTUS_DEBUG_SCOPE='" << scope
                  << "' has no valid component (backend, frontend, communicator, all). Ignoring.\n";
        return;
    }

    const LogLevel selected_level = root_level;
    const LogLevel non_selected_level = INFO_LOG_LEVEL;

    Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS")).setLogLevel(
        backend ? selected_level : non_selected_level);
    Logger::getInstance(LOG4CPLUS_TEXT("Backend")).setLogLevel(
        backend ? selected_level : non_selected_level);
    Logger::getInstance(LOG4CPLUS_TEXT("Process")).setLogLevel(
        backend ? selected_level : non_selected_level);

    Logger::getInstance(LOG4CPLUS_TEXT("Frontend")).setLogLevel(
        frontend ? selected_level : non_selected_level);

    Logger::getInstance(LOG4CPLUS_TEXT("UcxCommunicator")).setLogLevel(
        communicator ? selected_level : non_selected_level);
    Logger::getInstance(LOG4CPLUS_TEXT("TcpCommunicator")).setLogLevel(
        communicator ? selected_level : non_selected_level);
}

void rootLoggerConfig() {
    // Create console appender with a pattern layout
    SharedAppenderPtr consoleAppender(new ConsoleAppender());
    consoleAppender->setName(LOG4CPLUS_TEXT("console"));

    // Define the pattern layout string
    std::string pattern = "%D{%H:%M:%S.%q} [%-5p] [%c:%L] [%M]:%n"
                          "                       %m%n";
    // %D = date/time (with milliseconds via %q)
    // %p = log level (%-5p pads to 5 chars)
    // %c = logger name
    // %F = file name
    // %L = line number
    // %M = function name
    // %m = log message
    // %n = newline

    consoleAppender->setLayout(std::unique_ptr<Layout>(new PatternLayout(LOG4CPLUS_TEXT(pattern))));

    // Get the root logger
    Logger root = Logger::getRoot();
    root.removeAllAppenders();          // Remove any default appender
    root.addAppender(consoleAppender);  // Add our configured one

    // Set log level from env
    std::string logLevelString = getEnvVar("GVIRTUS_LOGLEVEL");
    LogLevel logLevel = INFO_LOG_LEVEL;
    if (!logLevelString.empty()) {
        try {
            logLevel = static_cast<LogLevel>(std::stoi(logLevelString));
        } catch (const std::exception& e) {
            std::cerr << "[GVIRTUS WARNING] Invalid GVIRTUS_LOGLEVEL value: '" << logLevelString
                      << "'. Using default INFO_LOG_LEVEL. (" << e.what() << ")\n";
            logLevel = INFO_LOG_LEVEL;
        }
    }

    root.setLogLevel(logLevel);
    apply_debug_component_scope(logLevel);
}

int main(int argc, char** argv) {
    rootLoggerConfig();

    // Get the main logger (GVirtuS); other loggers will inherit unless overridden
    logger = Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS"));

    LOG4CPLUS_INFO(logger, "GVirtuS backend: 0.0.12 version");

    std::string config_path;
#ifdef _CONFIG_FILE_JSON
    config_path = _CONFIG_FILE_JSON;
#endif
    config_path = (argc == 2) ? std::string(argv[1]) : std::string("");

    LOG4CPLUS_INFO(logger, "Configuration: " << config_path);

    // FIXME: Try - Catch? No.
    try {
        gvirtus::backend::Backend backend(config_path);

        LOG4CPLUS_INFO(logger, "[Process " << getpid() << "] Up and running!");
        backend.Start();
    } catch (const std::exception& e) {
        LOG4CPLUS_ERROR(logger, "Exception:" << e.what());
    }

    LOG4CPLUS_INFO(logger, "[Process " << getpid() << "] Shutdown");
    return 0;
}
