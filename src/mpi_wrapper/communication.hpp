#pragma once

#include "getters.hpp"
#include "structs.hpp"
#include "types.hpp"
#include <mpi.h>

namespace MPIw {

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

template <typename T>
void Bcast(MPI_Comm comm, std::vector<T>& data, int count, int root) {
    if (Comm_rank(comm) != root)
        data.resize(count);

    MPI_Bcast(data.data(), count, types::get_mpi_type<T>(), root, comm);
}

template <typename T>
void Bcast_send(MPI_Comm comm, const std::vector<T>& data) {
    MPI_Bcast(data.data(), data.size(), types::get_mpi_type<T>(),
              Comm_rank(comm), comm);
}

template <typename T>
std::vector<T> Bcast_recv(MPI_Comm comm, int count, int root) {
    std::vector<T> out(count);
    MPI_Bcast(out.data(), count, types::get_mpi_type<T>(), root, comm);
    return out;
}

template <typename T>
void Bcast_send_managed(MPI_Comm comm, const std::vector<T>& data) {
    int count = data.size();
    int my_rank = Comm_rank(comm);
    MPI_Bcast(&count, 1, MPI_INT, my_rank, comm);
    BCaset_send(comm, data);
}

template <typename T>
std::vector<T> Bcast_recv_managed(MPI_Comm comm, int root) {
    int count;
    MPI_Bcast(&count, 1, MPI_INT, root, comm);
    return Bcast_recv<T>(comm, count, root);
}
} // namespace MPIw