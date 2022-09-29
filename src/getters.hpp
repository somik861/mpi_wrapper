#pragma once
#include "types.hpp"
#include <array>
#include <iterator>
#include <mpi.h>
#include <string>

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
    errors::error_message(MPI_Comm_size(comm, &size));
    return size;
}

inline int Group_rank(MPI_Group group) {
    int rank;
    errors::error_message(MPI_Group_rank(group, &rank));
    return rank;
}

inline int Group_size(MPI_Group group) {
    int size;
    errors::error_message(MPI_Group_size(group, &size));
    return size;
}

inline std::string Get_processor_name() {
    std::array<char, MPI_MAX_PROCESSOR_NAME> name;
    int count;

    errors::error_message(MPI_Get_processor_name(name.begin(), &count));
    return std::string(name.begin(), name.begin() + count);
}
} // namespace MPIw