#include <iostream>
#include <cstdlib>
#include <unistd.h>

int main(int argc, char* argv[]) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    std::cout << "Hello, HPC World!" << std::endl;
    std::cout << "Running on host: " << hostname << std::endl;

    if (argc > 1) {
        std::cout << "Arguments passed:" << std::endl;
        for (int i = 1; i < argc; ++i) {
            std::cout << "  [" << i << "] " << argv[i] << std::endl;
        }
    }

    return 0;
}
