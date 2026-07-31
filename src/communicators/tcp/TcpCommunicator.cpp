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
 * @file   TcpCommunicator.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 8 12:08:33 2009
 *
 * @brief
 *
 *
 */

#include "TcpCommunicator.h"
#include <cstdio>
#include <netinet/tcp.h>

#include <sys/uio.h>

#ifndef _WIN32

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#else
#include <WinSock2.h>
static bool initialized = false;
#endif

#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Rdma.h>
#include <gvirtus/communicators/Endpoint_Tcp.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;
using gvirtus::communicators::TcpCommunicator;

TcpCommunicator::TcpCommunicator(const std::string &communicator) {
#ifdef _WIN32
    if (!initialized) {
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
            throw runtime_error("Cannot initialized WinSock.");
        initialized = true;
    }
#endif

    const char *valueptr = strstr(communicator.c_str(), "://") + 3;
    const char *portptr = strchr(valueptr, ':');
    if (portptr == NULL) throw runtime_error("Port not specified.");
    mPort = (short)strtol(portptr + 1, NULL, 10);

#ifdef _WIN32
    char *hostname = _strdup(valueptr);
#else
    char *hostname = strdup(valueptr);
#endif

    hostname[portptr - valueptr] = 0;
    mHostname = string(hostname);
    struct hostent *ent = gethostbyname(hostname);
    free(hostname);
    if (ent == NULL)
        throw runtime_error("TcpCommunicator: Can't resolve hostname '" + mHostname + "'.");
    mInAddrSize = ent->h_length;
    mInAddr = new char[mInAddrSize];
    memcpy(mInAddr, *ent->h_addr_list, mInAddrSize);
}

TcpCommunicator::TcpCommunicator(const char *hostname, short port) {
    mHostname = string(hostname);
    struct hostent *ent = gethostbyname(hostname);
    if (ent == NULL)
        throw runtime_error("TcpCommunicator: Can't resolve hostname '" + mHostname + "'.");
    mInAddrSize = ent->h_length;
    mInAddr = new char[mInAddrSize];
    memcpy(mInAddr, *ent->h_addr_list, mInAddrSize);
    mPort = port;
}

TcpCommunicator::TcpCommunicator(int fd, const char *hostname) {
    mSocketFd = fd;
    mHostname = hostname ? std::string(hostname) : "unknown";
    InitializeStream();
}

TcpCommunicator::~TcpCommunicator() {
    //    close(mSocketFd);
    delete[] mInAddr;
}

void TcpCommunicator::Serve() {
    LOG4CPLUS_DEBUG(logger, "Serve() called");

    struct sockaddr_in socket_addr;

    if ((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        throw runtime_error("TcpCommunicator: Can't create socket: " + string(strerror(errno)) +
                            ".");

    memset((char *)&socket_addr, 0, sizeof(struct sockaddr_in));

    socket_addr.sin_family = AF_INET;
    socket_addr.sin_port = htons(mPort);
    socket_addr.sin_addr.s_addr = INADDR_ANY;

    char on = 1;
    setsockopt(mSocketFd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    int bindResult = bind(mSocketFd, (struct sockaddr *)&socket_addr, sizeof(struct sockaddr_in));
    if (bindResult != 0)
        throw runtime_error("TcpCommunicator: Can't bind socket: " + string(strerror(errno)) + ".");

    int listenResult = listen(mSocketFd, 5);
    if (listenResult != 0)
        throw runtime_error(
            "TcpCommunicator: Can't listen from socket: " + string(strerror(errno)) + ".");

    LOG4CPLUS_INFO(logger, "Listening on port " << mPort);
}

const gvirtus::communicators::Communicator *const TcpCommunicator::Accept() const {
    LOG4CPLUS_TRACE(logger, "Accept() waiting for connection...");

    unsigned client_socket_fd;
    struct sockaddr_in client_socket_addr;
#ifndef _WIN32
    unsigned client_socket_addr_size;
#else
    int client_socket_addr_size;
#endif

    client_socket_addr_size = sizeof(struct sockaddr_in);
    if ((client_socket_fd =
             accept(mSocketFd, (sockaddr *)&client_socket_addr, &client_socket_addr_size)) == 0 ||
        errno == EINTR) {
        return nullptr;
    }

    const char *client_ip = inet_ntoa(client_socket_addr.sin_addr);
    int client_port = ntohs(client_socket_addr.sin_port);
    LOG4CPLUS_INFO(logger, "Client connected from " << client_ip << ":" << client_port
                           << " (fd=" << client_socket_fd << ")");
    return new TcpCommunicator(client_socket_fd, client_ip);
}

void TcpCommunicator::Connect() {
    LOG4CPLUS_DEBUG(logger, "Connect() to " << mHostname << ":" << mPort);

    struct sockaddr_in remote;

    if ((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        throw runtime_error("TcpCommunicator: Can't create socket: " + string(strerror(errno)) +
                            ".");

    remote.sin_family = AF_INET;
    remote.sin_port = htons(mPort);
    memcpy(&remote.sin_addr, mInAddr, mInAddrSize);

    if (connect(mSocketFd, (struct sockaddr *)&remote, sizeof(struct sockaddr_in)) != 0)
        throw runtime_error("TcpCommunicator: Can't connect to socket: " + string(strerror(errno)) +
                            ".");

    InitializeStream();

    LOG4CPLUS_INFO(logger, "Connected to " << mHostname << ":" << mPort);
}

void TcpCommunicator::Close() {}

size_t TcpCommunicator::Read(char *buffer, size_t size) {
    LOG4CPLUS_TRACE(logger, "Read() size=" << size);

    mpInput->read(buffer, size);

    size_t ret_value;
    if (mpInput->bad() || mpInput->eof())
        ret_value = 0;
    else
        ret_value = size;

    if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
        std::ostringstream hex;
        for (unsigned int i = 0; i < ret_value; i++)
            hex << i << " READ " << std::uppercase << std::hex
                << std::setw(2) << std::setfill('0')
                << (0xFF & (unsigned int)buffer[i]) << "\n";
        LOG4CPLUS_TRACE(logger, "Read() returned " << ret_value << " bytes:\n" << hex.str());
    }
    return ret_value;
}

size_t TcpCommunicator::Write(const char *buffer, size_t size) {
    LOG4CPLUS_TRACE(logger, "Write() size=" << size);

    mpOutput->write(buffer, size);

    if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
        std::ostringstream hex;
        for (unsigned int i = 0; i < size; i++)
            hex << i << " WRITTEN " << std::uppercase << std::hex
                << std::setw(2) << std::setfill('0')
                << (0xFF & (unsigned int)buffer[i]) << "\n";
        LOG4CPLUS_TRACE(logger, "Write() returned " << size << " bytes:\n" << hex.str());
    }
    return size;
}

size_t TcpCommunicator::WriteIov(const struct iovec *iov, size_t iov_count) {
    /* Sin estacionamiento. El stream ya bufferiza; anadir por encima un vector del tamano
     * del mensaje entero solo suma un zero-fill, una copia y un mmap/munmap por llamada.
     *
     * No se toca nada mas del camino TCP: mismo stream, mismo Sync(), mismo formato de
     * cable. El objetivo es volver a lo que hacia el base, no mejorarlo. */
    if (iov == nullptr || iov_count == 0) return 0;
    size_t total = 0;
    for (size_t i = 0; i < iov_count; ++i) {
        if (iov[i].iov_len == 0) continue;
        mpOutput->write(static_cast<const char *>(iov[i].iov_base),
                        static_cast<std::streamsize>(iov[i].iov_len));
        total += iov[i].iov_len;
    }
    return total;
}

void TcpCommunicator::Sync() { mpOutput->flush(); }

namespace {
/* Afinado opcional del socket, tras GVIRTUS_TCP_TUNED=1. Apagado por defecto: el brazo
 * `modified_tcp` tiene que seguir midiendo el transporte tal como existe. */
bool gvs_tcp_tuned() {
    static const bool on = []() {
        const char *v = std::getenv("GVIRTUS_TCP_TUNED");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

void gvs_tune_socket(int fd) {
    if (fd < 0 || !gvs_tcp_tuned()) return;
    int on = 1;
    /* Nagle retiene segmentos pequenos esperando llenar un MSS. Con miles de RPC de
     * peticion/respuesta por batch, cada retencion cuesta una ida y vuelta. */
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
    /* Los tamanos por defecto no cubren el producto ancho-de-banda x retardo de 100 GbE:
     * el emisor se bloquea esperando ACK con capacidad de sobra en el cable. */
    int buf = 16 * 1024 * 1024;
    ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
    ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
    std::fprintf(stderr, "[GVS TCP] afinado: TCP_NODELAY + SO_SNDBUF/SO_RCVBUF=%d\n", buf);
    std::fflush(stderr);
}
}  // namespace

void TcpCommunicator::InitializeStream() {
#ifdef _WIN32
    FILE *i = _fdopen(mSocketFd, "r");
    FILE *o = _fdopen(mSocketFd, "w");
    mpInputBuf = new filebuf(i);
    mpOutputBuf = new filebuf(o);
#else
    /* Tercer argumento = tamano del bufer. Sin el, `stdio_filebuf` usa BUFSIZ (8 KB), lo
     * que trocea las transferencias grandes en miles de escrituras. */
    const size_t bufsz = gvs_tcp_tuned() ? (1u << 20) : static_cast<size_t>(BUFSIZ);
    gvs_tune_socket(mSocketFd);
    mpInputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd, ios_base::in, bufsz);
    mpOutputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd, ios_base::out, bufsz);
#endif

    mpInput = new istream(mpInputBuf);
    mpOutput = new ostream(mpOutputBuf);
}

extern "C" std::shared_ptr<TcpCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    std::string arg =
        "tcp://" + std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Tcp>(end)->address() +
        ":" +
        std::to_string(
            std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Tcp>(end)->port());
    return std::make_shared<TcpCommunicator>(arg);
}
