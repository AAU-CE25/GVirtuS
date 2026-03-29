/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written By: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 *
 * Edited By: Mariano Aponte <aponte2001@gmail.com>,
 *            Department of Science and Technologies, University of Naples Parthenope
 *            Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            Department of Computer Science, University College Dublin
 */

#include "gvirtus/backend/Backend.h"

#include <gvirtus/communicators/CommunicatorFactory.h>
#include <gvirtus/communicators/EndpointFactory.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>

using gvirtus::backend::Backend;

Backend::Backend(const fs::path &path) {
    // logger setup
    this->logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Backend"));

    // json setup
    if (not(fs::exists(path) and fs::is_regular_file(path) and path.extension() == ".json")) {
        LOG4CPLUS_ERROR(logger, " json path error: no such file.");
        exit(EXIT_FAILURE);
    }

    LOG4CPLUS_DEBUG(logger, " Json file has been loaded.");

    // endpoints setup
    LOG4CPLUS_TRACE(logger, "Initializing endpoints setup");

    _properties = common::JSON<Property>(path).parser();
    _children.reserve(_properties.endpoints());

    LOG4CPLUS_TRACE(logger, "Got properties and reserved children array");

    if (_properties.endpoints() > 1)
        LOG4CPLUS_INFO(logger,
                       "Application serves on " << _properties.endpoints() << " several endpoint");

    try {
        for (int i = 0; i < _properties.endpoints(); i++) {
            LOG4CPLUS_DEBUG(logger, "Setting up endpoint " << i << "/" << _properties.endpoints());

            auto secure = _properties.secure();
            LOG4CPLUS_TRACE(logger, "  secure = " << secure);

            auto endpoint = communicators::EndpointFactory::get_endpoint(path);
            LOG4CPLUS_TRACE(logger, "  endpoint created");

            auto communicator =
                communicators::CommunicatorFactory::get_communicator(endpoint, secure);
            LOG4CPLUS_TRACE(logger, "  communicator created");

            auto plugins = _properties.plugins().at(i);
            LOG4CPLUS_TRACE(logger, "  plugins loaded (" << plugins.size() << " entries)");

            _children.push_back(std::make_unique<Process>(communicator, plugins));
            LOG4CPLUS_DEBUG(logger, "Endpoint " << i << " ready");
        }
    } catch (const std::exception &e) {
        LOG4CPLUS_ERROR(logger, "Exception during process setup: " << e.what());
    }

    LOG4CPLUS_INFO(logger, "Backend Initialization is complete!");
}

void Backend::Start() {
    // std::function<void(std::unique_ptr<gvirtus::Thread> & children)> task =
    // [this](std::unique_ptr<gvirtus::Thread> &children) {
    //   LOG4CPLUS_DEBUG(logger, "[Thread " << std::this_thread::get_id() <<
    //   "]: Started."); children->Start(); LOG4CPLUS_DEBUG(logger, "✓ - [Thread "
    //   << std::this_thread::get_id() << "]: Finished.");
    // };
    LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "] " << "Backend::Start() called.");

    int pid = 0;

    // _children definition: "std::vector<std::unique_ptr<Process>> _children"
    for (int i = 0; i < _children.size(); i++) {
        activeChilds++;
        if ((pid = fork()) == 0) {
            _children[i]->Start();
            LOG4CPLUS_TRACE(logger, "Child exited.");
            break;
        }
    }

    /* PARENT */
    pid_t pid_wait = 0;
    int stat_loc;
    if (pid != 0) {
        signal(SIGINT, SIG_IGN);
        signal(SIGHUP, SIG_IGN);

        LOG4CPLUS_TRACE(logger, "Active childs: " << activeChilds);

        int status;
        do {
            LOG4CPLUS_DEBUG(
                logger, "[Process " << getpid() << "] "
                                    << "Waiting for childs to terminate. Current active childs: "
                                    << activeChilds);
            int waitres = wait(&status);
            activeChilds--;

            LOG4CPLUS_TRACE(logger, "Active childs: %d" << activeChilds);

            if (waitres < 0) {
                LOG4CPLUS_TRACE(logger, "Error " << strerror(errno) << " on wait.");
            } else {
                LOG4CPLUS_TRACE(logger, "Process " << waitres << " returned successfully.");
                break;
            }
        } while (not WIFEXITED(status) and not WIFSIGNALED(status));

        LOG4CPLUS_INFO(
            logger,
            "No child processes are currently running. Use CTRL + C to terminate the backend.");

        signal(SIGINT, sigint_handler);
        pause();
    }

    LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "] " << "Backend::Start() returned.");
}

void Backend::EventOccurred(std::string &event, void *object) {
    LOG4CPLUS_DEBUG(logger, "EventOccurred: " << event);
}
