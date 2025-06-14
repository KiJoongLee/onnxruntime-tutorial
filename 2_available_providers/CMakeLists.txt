cmake_minimum_required(VERSION 3.10.0)
project(available_providers VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME REQUIRED IMPORTED_TARGET libonnxruntime)

include_directories(
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

link_directories(
    ${ONNXRUNTIME_LIBRARY_DIRS}
)

add_executable(available_providers
    main.cpp
)

target_link_libraries(available_providers
    ${ONNXRUNTIME_LIBRARIES}
)
