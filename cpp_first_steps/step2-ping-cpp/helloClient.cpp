/*
 * helloClient.cpp — Step 2 (Plain TCP, no RDMA)
 *
 * Connects to helloServer, sends "PING", and prints the reply.
 *
 * Usage:
 *   ./helloClient <server_ip>
 *   ./helloClient es-dpu-01          ← use hostname if DNS resolves it
 *
 * C++ concepts introduced here:
 *   - argc / argv               : reading command-line arguments
 *   - std::string               : C++ string type
 *   - std::to_string()          : convert int → string
 *   - getaddrinfo()             : resolves hostname OR IP address to a sockaddr
 *   - connect()                 : initiate TCP handshake (client side)
 *   - write() / read()          : send and receive bytes over the socket
 */

#include <iostream>       // std::cout, std::cerr
#include <string>         // std::string, std::to_string
#include <cstring>        // std::memset
#include <unistd.h>       // close(), read(), write()
#include <netdb.h>        // getaddrinfo(), freeaddrinfo(), addrinfo
#include <sys/socket.h>   // socket(), connect()
#include <netinet/in.h>   // htons()

constexpr int PORT     = 9000;    // must match helloServer.cpp
constexpr int BUF_SIZE = 256;

int main(int argc, char *argv[])
{
    // ── 1. Parse command-line arguments ─────────────────────────────────────
    //
    // argc = count of arguments (the program name counts as argv[0])
    // argv = array of C-strings: argv[0]="./helloClient", argv[1]="<ip>", …
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <server_ip_or_hostname>\n";
        std::cerr << "Example: " << argv[0] << " es-dpu-01\n";
        return 1;
    }

    // Store argv[1] as a std::string — safer and easier to use than char*
    const std::string server = argv[1];

    // ── 2. Resolve the server address ────────────────────────────────────────
    //
    // getaddrinfo() handles BOTH IP addresses ("192.168.1.5") and hostnames
    // ("es-dpu-01"). It returns a linked list of results in `res`.
    //
    // addrinfo hints tells getaddrinfo what kind of socket we want.
    addrinfo hints{};                       // zero-initialise
    hints.ai_family   = AF_INET;            // IPv4 only
    hints.ai_socktype = SOCK_STREAM;        // TCP

    addrinfo *res = nullptr;
    const std::string port_str = std::to_string(PORT);   // "9000"

    // getaddrinfo returns 0 on success, non-zero on failure
    if (getaddrinfo(server.c_str(), port_str.c_str(), &hints, &res) != 0) {
        std::cerr << "[client] Cannot resolve host: " << server << '\n';
        return 1;
    }

    // ── 3. Create the TCP socket ─────────────────────────────────────────────
    int sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock < 0) {
        std::perror("socket");
        freeaddrinfo(res);
        return 1;
    }

    // ── 4. Connect (TCP three-way handshake) ─────────────────────────────────
    //
    // connect() BLOCKS until the connection succeeds or fails.
    // It will fail if the server isn't listening yet — make sure to start
    // helloServer BEFORE running helloClient.
    std::cout << "[client] Connecting to " << server << ':' << PORT << " …\n";
    if (connect(sock, res->ai_addr, res->ai_addrlen) < 0) {
        std::perror("connect");
        freeaddrinfo(res);
        close(sock);
        return 1;
    }
    freeaddrinfo(res);   // free the address list — no longer needed after connect
    std::cout << "[client] Connected!\n";

    // ── 5. Send a PING message ───────────────────────────────────────────────
    //
    // write() sends raw bytes. We use a string literal as the payload.
    // std::strlen() counts bytes up to (but not including) the '\0'.
    const char *ping = "PING";
    // write() returns the number of bytes written, or -1 on error.
    if (write(sock, ping, std::strlen(ping)) < 0)
        std::perror("write");
    std::cout << "[client] Sent: \"" << ping << "\"\n";

    // ── 6. Read the reply ────────────────────────────────────────────────────
    //
    // read() fills buf and returns how many bytes were actually received.
    // ssize_t is a signed size type (like int but pointer-sized) —
    // the sign lets read() return -1 on error as well as byte counts.
    char buf[BUF_SIZE] = {};
    ssize_t n = read(sock, buf, BUF_SIZE - 1);
    if (n <= 0) {
        std::cerr << "[client] No reply received\n";
        close(sock);
        return 1;
    }
    buf[n] = '\0';                              // null-terminate so we can print it
    std::cout << "[client] Reply: \"" << buf << "\"\n";

    // ── 7. Clean up ──────────────────────────────────────────────────────────
    close(sock);
    std::cout << "[client] Done.\n";
    return 0;
}
