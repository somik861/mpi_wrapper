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

class Init_raii {
  public:
    Init_raii(int* argc, char*** argv) { MPI_Init(argc, argv); }

    Init_raii(const Init_raii&) = delete;
    Init_raii& operator=(const Init_raii&) = delete;

    Init_raii(Init_raii&&) = delete;
    Init_raii&& operator=(Init_raii&&) = delete;

    ~Init_raii() { MPI_Finalize(); }
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