# Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
#             School of Computer Science, University College Dublin

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

# Find GoogleTest package (make sure it is installed or handled otherwise)
find_package(GTest QUIET)
if (NOT GTest_FOUND)
    message(STATUS "GTest not found, fetching with FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    )
    FetchContent_MakeAvailable(googletest)
    # After FetchContent_MakeAvailable, GTest targets become available, 
    # but variables like GTest_INCLUDE_DIRS might not be set, so you may want to set them manually:
    set(GTEST_INCLUDE_DIRS ${googletest_SOURCE_DIR}/include)
    set(GTEST_LIBRARIES gtest gtest_main)
endif()

# Add your CUDA source files manually
set(TEST_SOURCES
    test_cudart.cu
    test_nvrtc.cu
    test_curand.cu
    test_cublas.cu
    test_cufft.cu
    test_cudnn.cu
    test_cudadr.cu
    test_cusolver.cu
    test_cusparse.cu
)

## Alternatively, you can use file(GLOB ...) to automatically find all .cu files in a directory
# file(GLOB TEST_SOURCES "*.cu")
# But this way you cannot control the order of tests, so if a test crashes the backend, all the others will be failed too.

# Loop over source files and create separate targets
foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    
    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_include_directories(${TEST_NAME} PRIVATE
        ${GTEST_INCLUDE_DIRS}
    )

    target_link_libraries(${TEST_NAME} PRIVATE
        GTest::GTest
        GTest::Main
        cuda # This is cudaDr library
        curand
        cudart
        cublas
        cublasLt
        cufft
        cudnn
        cusolver
        cusparse
        nvrtc
    )

    # Register the test with ctest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# ===== Fake libcudart shim (see tests/fake_cudart.c) =====
# Lets Add()'s auto-detect GpuRef path be exercised for real by
# test_buffer_iov / test_protocol_loopback even on machines/CI runners with
# no GPU or CUDA toolkit: any pointer obtained via its cudaMalloc is
# registered, so its cudaPointerGetAttributes correctly reports Device for
# them and Unregistered for everything else, and cudaMemcpy does a real
# memcpy (in this fake environment "device" memory is just process memory).
# Without it, IsDevicePointer degrades to "always false" (libcudart
# unavailable) and both tests still pass, verifying the safe fallback path
# instead — see the printed message each test emits in that case.
add_library(fake_cudart SHARED fake_cudart.c)
set_target_properties(fake_cudart PROPERTIES
    OUTPUT_NAME cudart
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fake_cudart
    # dlopen tries "libcudart.so.12" first (DeviceMemory.cpp's candidate
    # list) — give the built .so that exact SONAME/filename via a symlink
    # rather than relying on the linker's default libcudart.so naming.
)
add_custom_command(TARGET fake_cudart POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:fake_cudart>
            ${CMAKE_CURRENT_BINARY_DIR}/fake_cudart/libcudart.so.12
)

# ===== Pure-C++ unit test: Buffer IoV (scatter/gather) extensions =====
# No CUDA, GPU, GTest, or running backend required — compiles Buffer.cpp
# directly and asserts the AddRef-vs-Add wire-format-identical invariant,
# plus Add()'s GpuRef auto-detect via DeviceMemory.cpp.
add_executable(test_buffer_iov
    test_buffer_iov.cpp
    ${PROJECT_SOURCE_DIR}/src/communicators/Buffer.cpp
    ${PROJECT_SOURCE_DIR}/src/communicators/DeviceMemory.cpp
)
set_target_properties(test_buffer_iov PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(test_buffer_iov PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_buffer_iov PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(test_buffer_iov fake_cudart)
add_test(NAME test_buffer_iov COMMAND test_buffer_iov)
set_tests_properties(test_buffer_iov PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fake_cudart"
)

# ===== Pure-C++ unit test: unified RPC protocol (framing + codec) =====
# Loopback Communicator drives a full request/response round trip through the
# base-class framing and the communicators::am envelope codec. No GPU needed.
# Also exercises a GpuRef-tagged response through Communicator's GPU-safe
# default WriteIov fallback.
add_executable(test_protocol_loopback
    test_protocol_loopback.cpp
    ${PROJECT_SOURCE_DIR}/src/communicators/Buffer.cpp
    ${PROJECT_SOURCE_DIR}/src/communicators/RpcCodec.cpp
    ${PROJECT_SOURCE_DIR}/src/communicators/DeviceMemory.cpp
)
set_target_properties(test_protocol_loopback PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(test_protocol_loopback PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_protocol_loopback PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(test_protocol_loopback fake_cudart)
add_test(NAME test_protocol_loopback COMMAND test_protocol_loopback)
set_tests_properties(test_protocol_loopback PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/fake_cudart"
)