cmake_minimum_required(VERSION 3.1) 
project(beam-cuda-miner LANGUAGES CXX CUDA)
set(TARGET_NAME beam-cuda-miner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "-arch=sm_50")

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED COMPONENTS system thread)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
endif()

if (MSVC)

    # use static runtime
    foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        flag_var CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)

        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)

    # use static runtime for cuda
    foreach(flag_var CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_DEBUG
        CMAKE_CUDA_FLAGS_RELEASE 
        CMAKE_CUDA_FLAGS_RELWITHDEBINFO
        CMAKE_CUDA_FLAGS_MINSIZEREL)

        if(${flag_var} MATCHES "-MD")
            string(REGEX REPLACE "-MD" "-MT" ${flag_var} "${${flag_var}}")
        endif()

    endforeach(flag_var)

    set(OPENSSL_MSVC_STATIC_RT ON)
    
endif()

set(OPENSSL_USE_STATIC_LIBS ON)

find_package(OpenSSL REQUIRED)
if( OpenSSL_FOUND )
    link_directories(${OPENSSL_LIBRARIES})
    include_directories(${OPENSSL_INCLUDE_DIR})
    message(STATUS "Using OpenSSL ${OPENSSL_VERSION}")
else()
    message(FATAL_ERROR "error, OpenSSL not found, please install: 'apt-get install libssl-dev' on Ubuntu")
endif()

set (HEADERS
    beam/utility/common.h
    beam/core/difficulty.h
)

set(MINER_SRC
    beamStratum.cpp
    cudaHost.cu
    main.cpp
    crypto/sha256.c
    beam/core/difficulty.cpp
    beam/core/uintBig.cpp
    beam/utility/common.cpp
)

add_executable(${TARGET_NAME} ${MINER_SRC} ${HEADERS}  )

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/beam)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/clHeaders)

if(UNIX)
    find_package(Threads)
    target_link_libraries(${TARGET_NAME} Threads::Threads)
    target_link_libraries(${TARGET_NAME} -static-libstdc++ -static-libgcc)
endif()

if(MSVC)
    target_link_libraries(${TARGET_NAME} crypt32.lib)
    if(MSVC_VERSION GREATER_EQUAL 1900)
        target_link_libraries(${TARGET_NAME} legacy_stdio_definitions.lib)
    endif()
endif()

#target_link_libraries(${TARGET_NAME} OpenCL::OpenCL)
target_link_libraries(${TARGET_NAME} ${OPENSSL_SSL_LIBRARY} ${OPENSSL_CRYPTO_LIBRARY})
target_link_libraries(${TARGET_NAME} ${Boost_LIBRARIES})
