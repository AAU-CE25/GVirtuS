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
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   TcpCommunicator.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 8 12:08:33 2009
 *
 * @brief
 *
 *
 */

#pragma once

#ifdef _WIN32
#include <fstream>
#else
#include <ext/stdio_filebuf.h>
#endif

#include "gvirtus/communicators/Communicator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

namespace gvirtus::communicators {
/**
 * TcpCommunicator implements a Communicator for the TCP/IP socket.
 */
class TcpCommunicator : public Communicator {
   public:
    TcpCommunicator() = default;
    TcpCommunicator(const std::string &communicator);
    TcpCommunicator(const char *hostname, short port);
    TcpCommunicator(int fd, const char *hostname);
    virtual ~TcpCommunicator();
    void Serve();
    const Communicator *const Accept() const;
    void Connect();
    size_t Read(char *buffer, size_t size);
    size_t Write(const char *buffer, size_t size);
    void Sync();
    void Close();

    std::string to_string() override { return "tcpcommunicator"; }
    std::string get_client_address() const { return mHostname; }

   private:
    log4cplus::Logger logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("TcpCommunicator"));
    void InitializeStream();
    std::istream *mpInput = nullptr;
    std::ostream *mpOutput = nullptr;
    std::string mHostname;
    char *mInAddr = nullptr;
    int mInAddrSize = 0;
    short mPort = 0;
    int mSocketFd = -1;
#ifdef _WIN32
    std::filebuf *mpInputBuf;
    std::filebuf *mpOutputBuf;
#else
    __gnu_cxx::stdio_filebuf<char> *mpInputBuf = nullptr;
    __gnu_cxx::stdio_filebuf<char> *mpOutputBuf = nullptr;
#endif
};
}  // namespace gvirtus::communicators
