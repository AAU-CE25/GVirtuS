# Step 1 – Hello HPC (C++ Intro)

A simple "Hello World" command-line program written in C++. Great starting point if you are new to C++.

---

## What the program does

- Prints `Hello, HPC World!` to the terminal.
- Detects and prints the **hostname** of the machine it is running on (useful when running on remote HPC nodes).
- Optionally prints any **command-line arguments** you pass to it.

---

## C++ concepts used

| Concept | Where |
|---------|-------|
| `#include` | Imports standard libraries (like `iostream` for printing) |
| `int main(int argc, char* argv[])` | Entry point — `argc` is the argument count, `argv` is the argument list |
| `std::cout` | Prints text to the terminal (`<<` chains things together) |
| `std::endl` | Ends the line (like pressing Enter) |
| `gethostname()` | A system call that returns the machine's hostname |

---

## Files

| File | Description |
|------|-------------|
| `helloHPC.cpp` | The C++ source code |
| `Makefile` | Build instructions — tells the compiler how to build the program |
| `helloHPC` | The compiled binary (created after running `make`) |

---

## How to build

```bash
make
```

This runs `g++ -std=c++17 -Wall -o helloHPC helloHPC.cpp` behind the scenes.

To remove the compiled binary:

```bash
make clean
```

---

## How to run

```bash
./helloHPC
```

Expected output:
```
Hello, HPC World!
Running on host: your-machine-name
```

You can also pass arguments:

```bash
./helloHPC foo bar
```

Output:
```
Hello, HPC World!
Running on host: your-machine-name
Arguments passed:
  [1] foo
  [2] bar
```
