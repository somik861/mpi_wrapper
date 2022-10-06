#pragma once

#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace MPIw::structs {
template <typename T>
struct Recv_st {
    T data;
    MPI_Status status;
};

} // namespace MPIw::structs