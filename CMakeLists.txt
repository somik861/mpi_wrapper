cmake_minimum_required(VERSION 3.18)
project(mpi_test C CXX)

set(CMAKE_CXX_STANDARD 20)
add_definitions(-fmax-errors=1)

find_package(MPI REQUIRED)
set(LIBS ${LIBS} MPI::MPI_CXX)

find_package(fmt CONFIG REQUIRED)
set(LIBS ${LIBS} fmt::fmt-header-only)

add_executable(main src/main.cpp)
add_executable(random_walk src/random_walk.cpp)
add_executable(run_tests tests/main.cpp)
target_link_libraries(main ${LIBS})
target_link_libraries(random_walk ${LIBS})
target_link_libraries(run_tests ${LIBS})

