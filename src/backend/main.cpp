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

void rootLoggerConfig() {
    // Create console appender with a pattern layout
    SharedAppenderPtr consoleAppender(new ConsoleAppender());
    consoleAppender->setName(LOG4CPLUS_TEXT("console"));

    // Define the pattern layout string
    std::string pattern = "%D{%H:%M:%S.%q} [%-5p] [%c] (%F:%L) - %m%n";
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
