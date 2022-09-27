#pragma once
#include <mpi.h>
#include "types.hpp"

namespace MPIw {
inline int Get_count(const MPI_Status& status, MPI_Datatype type) {
    int count;
    MPI_Get_count(&status, type, &count);
    return count;
}

template <typename T>
int Get_count(const MPI_Status& status) {
    return Get_count(status, types::get_mpi_type<T>());
}

inline int Comm_rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

inline int Comm_size(MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}
} // namespace MPIw