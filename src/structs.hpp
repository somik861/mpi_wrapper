#pragma once

#include <mpi.h>
#include <vector>

namespace MPIw::structs {
template <typename T>
struct Recv_st {
    std::vector<T> data;
    MPI_Status status;
};
} // namespace MPIw::structs