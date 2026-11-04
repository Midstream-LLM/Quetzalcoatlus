cmake_minimum_required(VERSION 3.10)

project(quetzal VERSION 0.1)
include(CTest)

message("CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set compile option
if (NOT MSVC)
    add_compile_options(-fPIC -Wshadow -Wall)
else()
    add_compile_options(/W4 /utf-8)
endif()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(amd64)")
    if (MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2 -mfma)
    endif()
endif()

if (APPLE)
    add_compile_options(-Wsign-compare)
    add_compile_options(-mmacosx-version-min=15)
endif()

# find openmp
find_package(OpenMP)

add_library(quetzal STATIC
            ${CMAKE_SOURCE_DIR}/lib/model/riverstrike/attention.cpp
            ${CMAKE_SOURCE_DIR}/lib/model/riverstrike/kv_cache.cpp
            ${CMAKE_SOURCE_DIR}/lib/model/riverstrike/transformer.cpp
            ${CMAKE_SOURCE_DIR}/lib/model/riverstrike/riverstrike.cpp
            ${CMAKE_SOURCE_DIR}/lib/model/qgpt_bin_reader.cpp
            ${CMAKE_SOURCE_DIR}/lib/engine/attn.cpp
            ${CMAKE_SOURCE_DIR}/lib/engine/chat.cpp
            ${CMAKE_SOURCE_DIR}/lib/engine/experiment.cpp
            ${CMAKE_SOURCE_DIR}/lib/engine/performance.cpp
            ${CMAKE_SOURCE_DIR}/lib/engine/repetition_penalty.cpp
            ${CMAKE_SOURCE_DIR}/lib/engine/utils.cpp
            ${CMAKE_SOURCE_DIR}/lib/tensor/tensor.cpp
            ${CMAKE_SOURCE_DIR}/lib/thirdparty/wcwidth.cpp
            ${CMAKE_SOURCE_DIR}/lib/util/byte_to_unicode.cpp
            ${CMAKE_SOURCE_DIR}/lib/util/chat_message.cpp
            ${CMAKE_SOURCE_DIR}/lib/util/cli.cpp
            ${CMAKE_SOURCE_DIR}/lib/util/fs.cpp
            ${CMAKE_SOURCE_DIR}/lib/util/perf_info.cpp
            ${CMAKE_SOURCE_DIR}/lib/util/ppm.cpp
            ${CMAKE_SOURCE_DIR}/lib/util/timestamp.cpp
            ${CMAKE_SOURCE_DIR}/lib/util/utf8.cpp
            ${CMAKE_SOURCE_DIR}/lib/bbpe/tokenizer.cpp)
target_include_directories(quetzal PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    target_link_libraries(quetzal PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(quetzal PUBLIC QGPT_USE_OPENMP)
else()
    message(STATUS "OpenMP not found")
endif()

# build deploy
add_executable(deploy ${CMAKE_SOURCE_DIR}/deploy.cpp)
target_link_libraries(deploy quetzal)
# build test and tools
add_executable(thread_bench ${CMAKE_SOURCE_DIR}/test/thread_bench.cpp)
add_executable(tokenizer ${CMAKE_SOURCE_DIR}/tokenizer.cpp)
target_link_libraries(thread_bench quetzal)
target_link_libraries(tokenizer quetzal)

if(BUILD_TESTING)
    add_executable(kv_cache_verify ${CMAKE_SOURCE_DIR}/test/kv_cache_verify.cpp)
    target_link_libraries(kv_cache_verify quetzal)
    add_test(NAME kv_cache_verify COMMAND kv_cache_verify)

    add_executable(softmax_causal_verify ${CMAKE_SOURCE_DIR}/test/softmax_causal_verify.cpp)
    target_link_libraries(softmax_causal_verify quetzal)
    add_test(NAME softmax_causal_verify COMMAND softmax_causal_verify)

    add_executable(tensor_test ${CMAKE_SOURCE_DIR}/test/tensor_test.cpp)    
    target_link_libraries(tensor_test quetzal)
    add_test(NAME tensor_test COMMAND tensor_test)
endif()
