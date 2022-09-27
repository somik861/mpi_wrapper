#pragma once

#include "structs.hpp"
#include "types.hpp"
#include <mpi.h>

namespace MPIw {
namespace details {}

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

class Init_raii {
  public:
    Init_raii(int* argc, char*** argv) { MPI_Init(argc, argv); }

    Init_raii(const Init_raii&) = delete;
    Init_raii& operator=(const Init_raii&) = delete;

    Init_raii(Init_raii&&) = delete;
    Init_raii&& operator=(Init_raii&&) = delete;

    ~Init_raii() { MPI_Finalize(); }
};

class Comm_raii {
  public:
    MPI_Comm comm = MPI_COMM_NULL;

    Comm_raii() = default;
    Comm_raii(const Comm_raii&) = delete;
    Comm_raii& operator=(const Comm_raii&) = delete;

    Comm_raii(Comm_raii&&) = delete;
    Comm_raii&& operator=(Comm_raii&&) = delete;

    ~Comm_raii() { MPI_Comm_free(&comm); }
};

template <typename T>
structs::Recv_st<T>
Recv(MPI_Comm comm, int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) {
    structs::Recv_st<T> out;
    MPI_Status stat;
    MPI_Datatype type = types::get_mpi_type<T>();

    MPI_Probe(source, tag, comm, &stat);

    out.data.resize(Get_count<T>(stat));
    MPI_Recv(out.data.data(), out.data.size(), type, stat.MPI_SOURCE,
             stat.MPI_TAG, comm, &out.status);

    assert(stat.MPI_SOURCE == out.status.MPI_SOURCE);
    assert(stat.MPI_TAG == out.status.MPI_TAG);
    assert(out.data.size() == Get_count<T>(out.status));

    return out;
}

template <typename T>
void Send(MPI_Comm comm, const std::vector<T>& data, int dest, int tag) {
    MPI_Send(data.data(), data.size(), types::get_mpi_type<T>(), dest, tag,
             comm);
}
} // namespace MPIw