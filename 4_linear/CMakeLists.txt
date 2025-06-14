cmake_minimum_required(VERSION 3.10.0)
project(linear VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME REQUIRED IMPORTED_TARGET libonnxruntime)

include_directories(
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

link_directories(
    ${ONNXRUNTIME_LIBRARY_DIRS}
)

add_executable(linear
    main.cpp
)

target_link_libraries(linear
    ${ONNXRUNTIME_LIBRARIES}
)

add_executable(linear2
    main2.cpp
)

target_link_libraries(linear2
    ${ONNXRUNTIME_LIBRARIES}
)

add_executable(linear3
    main3.cpp
)

target_link_libraries(linear3
    ${ONNXRUNTIME_LIBRARIES}
)
