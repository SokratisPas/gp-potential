cmake_minimum_required(VERSION 3.10)

project(gp-potential VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(
    gp-potential
    src/main.cpp
)

target_link_libraries(gp-potential PRIVATE MPI::MPI_CXX)

target_include_directories(gp-potential PRIVATE ${MPI_CXX_INCLUDE_DIRS})
