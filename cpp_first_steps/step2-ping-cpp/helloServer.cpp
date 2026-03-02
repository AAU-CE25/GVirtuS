/*
 * helloServer.cpp — Step 2 (Plain TCP, no RDMA)
 *
 * Waits for a client to connect, reads a "PING" message,
 * and replies "Hello from <hostname>".
 *
 * C++ concepts introduced here:
 *   - std::string               : safer than raw char arrays
 *   - gethostname()             : POSIX system call to get the machine name
 *   - socket / bind / listen    : standard TCP server boilerplate (POSIX C API)
 *   - int vs ssize_t            : why we cast when comparing read() return values
 *   - RAII-style cleanup        : closing fds before exit
 */

#include <iostream>       // std::cout, std::cerr
#include <string>         // std::string
#include <cstring>        // std::memset, std::strlen
#include <unistd.h>       // close(), gethostname(), read(), write()
#include <netinet/in.h>   // sockaddr_in, htons(), INADDR_ANY
#include <sys/socket.h>   // socket(), bind(), listen(), accept()

// The TCP port both programs agree on (must match helloClient.cpp)
constexpr int PORT     = 9000;
// Maximum bytes we will read in one call
constexpr int BUF_SIZE = 256;

int main()
{
    // ── 1. Get our own hostname ──────────────────────────────────────────────
    //
    // char array: a fixed-size block of characters (old C style, still common
    // when calling POSIX system functions that fill a buffer for you).
    char hostname[BUF_SIZE];
    gethostname(hostname, sizeof(hostname));   // fills hostname[] with machine name
    std::cout << "[server] Running on host: " << hostname << '\n';

    // ── 2. Create a TCP socket ───────────────────────────────────────────────
    //
    // socket() returns a file descriptor (just an int). Every open file,
    // pipe, or socket in Linux is identified by an int fd.
    //   AF_INET     = IPv4
    //   SOCK_STREAM = TCP (reliable, connection-oriented)
    //   0           = let the OS pick the right protocol (TCP here)
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::perror("socket");   // prints the OS error message
        return 1;
    }

    // Allow reuse of the port immediately after the program exits.
    // Without this you'd get "Address already in use" on quick restarts.
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // ── 3. Bind to a port ────────────────────────────────────────────────────
    //
    // sockaddr_in is a struct that holds an IPv4 address + port.
    // std::memset zeroes it out first so no garbage values sneak in.
    sockaddr_in addr{};                          // {} zero-initialises in C++
    addr.sin_family      = AF_INET;             // IPv4
    addr.sin_addr.s_addr = INADDR_ANY;          // accept connections on any interface
    addr.sin_port        = htons(PORT);         // htons = "host to network short"
                                                //   converts port to big-endian byte order

    if (bind(server_fd,
             reinterpret_cast<sockaddr *>(&addr),   // cast to generic socket addr
             sizeof(addr)) < 0) {
        std::perror("bind");
        return 1;
    }

    // ── 4. Listen for incoming connections ──────────────────────────────────
    //
    // The second argument (1) is the backlog — how many pending connections
    // the OS will queue while we haven't called accept() yet.
    if (listen(server_fd, 1) < 0) {
        std::perror("listen");
        return 1;
    }
    std::cout << "[server] Listening on port " << PORT << " …\n";

    // ── 5. Accept one client ─────────────────────────────────────────────────
    //
    // accept() BLOCKS until a client connects.
    // It returns a NEW file descriptor representing that specific connection.
    int client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) {
        std::perror("accept");
        return 1;
    }
    std::cout << "[server] Client connected!\n";

    // ── 6. Read the PING message ─────────────────────────────────────────────
    //
    // read() fills buf with up to BUF_SIZE-1 bytes.
    // We leave one byte free so we can add a '\0' (null terminator) and
    // safely treat the buffer as a C-string.
    char buf[BUF_SIZE] = {};                     // zero the buffer
    ssize_t n = read(client_fd, buf, BUF_SIZE - 1);
    if (n <= 0) {
        std::cerr << "[server] Failed to read from client\n";
        return 1;
    }
    buf[n] = '\0';                               // manually null-terminate
    std::cout << "[server] Received: \"" << buf << "\"\n";

    // ── 7. Send back "Hello from <hostname>" ────────────────────────────────
    //
    // std::string lets us concatenate with + (not possible with raw char[]).
    std::string reply = "Hello from " + std::string(hostname) + "!";

    // .c_str()  → gives a const char* pointer to the string's data
    // .size()   → number of characters (not counting the '\0')
    // write() returns the number of bytes written, or -1 on error.
    // We check it to avoid the -Wunused-result compiler warning.
    if (write(client_fd, reply.c_str(), reply.size()) < 0)
        std::perror("write");
    std::cout << "[server] Sent: \"" << reply << "\"\n";

    // ── 8. Clean up ──────────────────────────────────────────────────────────
    //
    // Always close file descriptors — the OS won't reuse the port until
    // the socket is fully closed (especially server_fd).
    close(client_fd);
    close(server_fd);
    std::cout << "[server] Done.\n";
    return 0;
}
