# Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
#             School of Computer Science, University College Dublin

project(gvirtus-plugin-cuda)

function(configure_cudadr_plugin)
    find_package(CUDA REQUIRED)

    find_path(CUDADR_INCLUDE_DIRECTORY
        cuda.h
        PATHS ${CUDA_INCLUDE_DIRS})

    if(NOT CUDADR_INCLUDE_DIRECTORY)
        message(WARNING "cuda.h not found, skipping plugins/cudadr")
        return()
    endif()

    message(STATUS "cudaDr include directory found at: ${CUDADR_INCLUDE_DIRECTORY}")

    include_directories(${CUDADR_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS})

    # Derive a fallback search path from the resolved CUDA toolkit cudart
    # location, so we can also look inside <toolkit>/lib64/stubs/. Without
    # nvidia-driver-* installed (typical dev container), the real
    # /usr/lib/x86_64-linux-gnu/libcuda.so.1 is missing; the toolkit ships
    # a link-only stub at lib64/stubs/libcuda.so. This change is from
    # Kajetan's fix/cudadr-stubs-version branch.
    get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)

    find_library(CUDADR_LIBRARY
        cuda
        PATHS
            ${CUDA_LIBRARIES_PATH}
            ${CUDA_LIBRARIES_PATH}/stubs
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs
            /usr/local/cuda/lib64/stubs)

    if(NOT CUDADR_LIBRARY)
        message(WARNING "cudaDr library not found, skipping plugins/cudadr")
        return()
    endif()

    message(STATUS "cudaDr library found at ${CUDADR_LIBRARY}, adding plugins/cudadr")

    set(CUDADR_VERSION "${CUDADR_LIBRARY}")
    while(IS_SYMLINK "${CUDADR_VERSION}")
        message(STATUS "${CUDADR_VERSION}")
        execute_process(COMMAND readlink "${CUDADR_VERSION}"
            OUTPUT_VARIABLE CUDADR_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endwhile()
    message(STATUS "Resolved CUDADR version: ${CUDADR_VERSION}")

    # Stub library is named just "libcuda.so" with no ".N" version suffix,
    # so the regex above leaves CUDADR_VERSION = path. Fall back to "1".
    if(CUDADR_VERSION MATCHES ".*\\.so\\.([0-9]+(\\.[0-9]+)*)")
        string(REGEX REPLACE ".*\\.so\\.([0-9]+(\\.[0-9]+)*)" "\\1" CUDADR_VERSION "${CUDADR_VERSION}")
    else()
        set(CUDADR_VERSION "1")
        message(STATUS "No version suffix found in libcuda.so, using default: ${CUDADR_VERSION}")
    endif()


    gvirtus_add_backend(cuda ${CUDADR_VERSION}
        backend/CudaDrHandler_context.cpp
        backend/CudaDrHandler_device.cpp
        backend/CudaDrHandler_driver_entry_point.cpp
        backend/CudaDrHandler_error.cpp
        backend/CudaDrHandler_event.cpp
        backend/CudaDrHandler_execution.cpp
        backend/CudaDrHandler_initialization.cpp
        backend/CudaDrHandler_memory.cpp
        backend/CudaDrHandler_module.cpp
        backend/CudaDrHandler_occupancy.cpp
        backend/CudaDrHandler_stream.cpp
        backend/CudaDrHandler_streammemory.cpp
        backend/CudaDrHandler_tensormap.cpp
        backend/CudaDrHandler_texture.cpp
        backend/CudaDrHandler_unified.cpp
        backend/CudaDrHandler_version.cpp
        backend/CudaDrHandler_virtmemory.cpp
        backend/CudaDrHandler.cpp
    )

    target_link_libraries(${PROJECT_NAME} ${CUDADR_LIBRARY})

    gvirtus_add_frontend(cuda ${CUDADR_VERSION}
        frontend/CudaDr_context.cpp
        frontend/CudaDr_device.cpp
        frontend/CudaDr_driver_entry_point.cpp
        frontend/CudaDr_error.cpp
        frontend/CudaDr_event.cpp
        frontend/CudaDr_execution.cpp
        frontend/CudaDr_initialization.cpp
        frontend/CudaDr_memory.cpp
        frontend/CudaDr_module.cpp
        frontend/CudaDr_occupancy.cpp
        frontend/CudaDr_stream.cpp
        frontend/CudaDr_streammemory.cpp
        frontend/CudaDr_tensormap.cpp
        frontend/CudaDr_texture.cpp
        frontend/CudaDr_unified.cpp
        frontend/CudaDr_version.cpp
        frontend/CudaDr_virtmemory.cpp
        frontend/CudaDrFrontend.cpp
        frontend/CudaDr_compat_stubs.cpp
    )
    # Link dl so cuGetExportTable can dlopen real libcuda (NVRTC-style passthrough).
    target_link_libraries(cuda dl)

    # CUDADR_VERSION resolves to "1" (libcuda.so.1 has no minor suffix), and
    # gvirtus_add_frontend's regex makes SOVERSION == VERSION. CMake then
    # skips the `libcuda.so.1 -> libcuda.so.1.minor` symlink and only
    # installs `libcuda.so`. That breaks nvcc-linked binaries that look up
    # libcuda.so.1 by SONAME. Force a minor so cmake builds the symlink
    # chain libcuda.so -> libcuda.so.1 -> libcuda.so.1.0.
    set_target_properties(cuda PROPERTIES
        VERSION   "${CUDADR_VERSION}.0"
        SOVERSION "${CUDADR_VERSION}")

    # Required by CudaDr_initialization.cpp to dlopen the real libcuda for the
    # cuInit reentrancy passthrough, avoiding recursive Frontend::Prepare via
    # libuct_cuda's module init.
    target_link_libraries(cuda dl)

endfunction()

# Call the function to configure cudadr plugin
configure_cudadr_plugin()