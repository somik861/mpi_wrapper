#pragma once
#include "types.hpp"
#include <array>
#include <iterator>
#include <mpi.h>
#include <string>

namespace MPIw {
inline int Get_count(
    const MPI_Status& status,
    MPI_Datatype type,
    const std::source_location& location = std::source_location::current()) {
    int count;
    errors::check_code(MPI_Get_count(&status, type, &count), location);
    return count;
}

template <typename T>
int Get_count(
    const MPI_Status& status,
    const std::source_location& location = std::source_location::current()) {
    return Get_count(status, types::get_mpi_type<T>(), location);
}

inline int Comm_rank(
    MPI_Comm comm,
    const std::source_location& location = std::source_location::current()) {
    int rank;
    errors::check_code(MPI_Comm_rank(comm, &rank), location);
    return rank;
}

inline int Comm_size(
    MPI_Comm comm,
    const std::source_location& location = std::source_location::current()) {
    int size;
    errors::check_code(MPI_Comm_size(comm, &size), location);
    return size;
}

inline int Group_rank(
    MPI_Group group,
    const std::source_location& location = std::source_location::current()) {
    int rank;
    errors::check_code(MPI_Group_rank(group, &rank), location);
    return rank;
}

inline int Group_size(
    MPI_Group group,
    const std::source_location& location = std::source_location::current()) {
    int size;
    errors::check_code(MPI_Group_size(group, &size), location);
    return size;
}

inline std::string Get_processor_name(
    const std::source_location& location = std::source_location::current()) {
    std::array<char, MPI_MAX_PROCESSOR_NAME> name;
    int count;

    errors::check_code(MPI_Get_processor_name(name.begin(), &count), location);
    return std::string(name.begin(), name.begin() + count);
}
} // namespace MPIw