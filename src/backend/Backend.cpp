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
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <vector>

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
        if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
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
        }
        else {
            for (int i = 0; i < _properties.endpoints(); i++) {
                LOG4CPLUS_DEBUG(logger, "Setting up endpoint " << i << "/" << _properties.endpoints());
                _children.push_back(std::make_unique<Process>(
                    communicators::CommunicatorFactory::get_communicator(
                        communicators::EndpointFactory::get_endpoint(path), _properties.secure()),
                    _properties.plugins().at(i)));
                    LOG4CPLUS_DEBUG(logger, "Endpoint " << i << " ready");
            }
        }
    } catch (const std::exception &e) {
        LOG4CPLUS_ERROR(logger, "Exception during process setup: " << e.what());
    }

    LOG4CPLUS_INFO(logger, "Backend Initialization is complete!");
}

void Backend::Start() {
    LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "] Backend::Start() called.");

    if (_children.empty()) {
        LOG4CPLUS_ERROR(logger, "No backend children/endpoints configured.");
        return;
    }

    auto start_child = [&](int index) -> pid_t {
        pid_t child_pid = fork();

        if (child_pid < 0) {
            LOG4CPLUS_ERROR(logger, "fork() failed for endpoint child " << index << ": "
                                                                      << strerror(errno));
            return -1;
        }

        if (child_pid == 0) {
            // This is the endpoint/server child. It owns Serve()/Accept().
            // Do not inherit the parent's signal handling policy when respawned.
            signal(SIGINT, SIG_DFL);
            signal(SIGHUP, SIG_DFL);

            LOG4CPLUS_INFO(logger, "[Process " << getpid()
                                               << "] Starting backend endpoint child "
                                               << index);

            _children[index]->Start();

            // If Process::Start() returns, the listening socket is gone. The parent
            // will detect this child exit and respawn it so the backend keeps
            // accepting future frontend connections.
            LOG4CPLUS_WARN(logger, "[Process " << getpid()
                                               << "] Backend endpoint child "
                                               << index
                                               << " returned from Process::Start().");

            // Avoid running parent-side C++ destructors in the forked child.
            _exit(EXIT_SUCCESS);
        }

        LOG4CPLUS_INFO(logger, "[Process " << getpid()
                                           << "] Spawned backend endpoint child "
                                           << child_pid
                                           << " for endpoint " << index);
        return child_pid;
    };

    std::vector<pid_t> child_pids(_children.size(), -1);
    activeChilds = 0;

    for (int i = 0; i < static_cast<int>(_children.size()); i++) {
        child_pids[i] = start_child(i);
        if (child_pids[i] > 0) {
            activeChilds++;
        }
    }

    signal(SIGINT, sigint_handler);
    signal(SIGHUP, SIG_IGN);

    // Parent supervisor loop. The old implementation waited for a child once and
    // then paused forever, which left the parent process alive but without any
    // listening socket when the endpoint child exited. Here we restart any endpoint
    // child that exits unexpectedly so the backend remains a persistent service.
    while (true) {
        LOG4CPLUS_DEBUG(logger, "[Process " << getpid()
                                            << "] Waiting for backend children. Active childs: "
                                            << activeChilds);

        int status = 0;
        pid_t dead_pid = wait(&status);

        if (dead_pid < 0) {
            if (errno == EINTR) {
                LOG4CPLUS_INFO(logger, "wait() interrupted. Stopping backend supervisor.");
                break;
            }

            if (errno == ECHILD) {
                LOG4CPLUS_WARN(logger, "No child processes found. Respawning endpoint children.");
                activeChilds = 0;
                for (int i = 0; i < static_cast<int>(_children.size()); i++) {
                    child_pids[i] = start_child(i);
                    if (child_pids[i] > 0) {
                        activeChilds++;
                    }
                }
                continue;
            }

            LOG4CPLUS_WARN(logger, "wait() failed: " << strerror(errno));
            sleep(1);
            continue;
        }

        int child_index = -1;
        for (int i = 0; i < static_cast<int>(child_pids.size()); i++) {
            if (child_pids[i] == dead_pid) {
                child_index = i;
                break;
            }
        }

        if (activeChilds > 0) {
            activeChilds--;
        }

        if (WIFEXITED(status)) {
            LOG4CPLUS_WARN(logger, "Backend child " << dead_pid
                                                    << " exited with status "
                                                    << WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            LOG4CPLUS_WARN(logger, "Backend child " << dead_pid
                                                    << " was killed by signal "
                                                    << WTERMSIG(status));
        } else {
            LOG4CPLUS_WARN(logger, "Backend child " << dead_pid
                                                    << " ended with unknown status.");
        }

        if (child_index < 0) {
            LOG4CPLUS_WARN(logger, "Unknown child " << dead_pid
                                                    << " exited. Not respawning.");
            continue;
        }

        LOG4CPLUS_WARN(logger, "Restarting backend endpoint child " << child_index);
        child_pids[child_index] = start_child(child_index);
        if (child_pids[child_index] > 0) {
            activeChilds++;
        }
    }

    LOG4CPLUS_INFO(logger, "Stopping backend children.");
    for (pid_t child_pid : child_pids) {
        if (child_pid > 0) {
            kill(child_pid, SIGINT);
        }
    }

    LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "] Backend::Start() returned.");
}

void Backend::EventOccurred(std::string &event, void *object) {
    LOG4CPLUS_DEBUG(logger, "EventOccurred: " << event);
}
