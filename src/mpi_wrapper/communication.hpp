#pragma once

#include "getters.hpp"
#include "structs.hpp"
#include "types.hpp"
#include <mpi.h>
#include <numeric>

namespace MPIw {

// ===================== BASIC Send/Recv =====================

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

// ===================== Bcast =====================

template <typename T>
std::vector<T>
Bcast(MPI_Comm comm, const std::vector<T>& data, int count, int root) {
    if (Comm_rank(comm) != root)
        return Bcast_recv<T>(comm, count, root);

    Bcast_send(comm, data);
    return {};
}

template <typename T>
std::vector<T>
Bcast_managed(MPI_Comm comm, const std::vector<T>& data, int count, int root) {
    if (Comm_rank(comm) != root)
        return Bcast_recv_managed<T>(comm, count, root);

    Bcast_send_managed(comm, data);
    return {};
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
    BCast_send(comm, data);
}

template <typename T>
std::vector<T> Bcast_recv_managed(MPI_Comm comm, int root) {
    int count;
    MPI_Bcast(&count, 1, MPI_INT, root, comm);
    return Bcast_recv<T>(comm, count, root);
}

// ===================== Gather =====================
template <typename T>
std::vector<T> Gather(MPI_Comm comm, const std::vector<T>& data, int root) {
    if (root == Comm_rank(comm))
        return Gather_recv(comm, data);

    Gather_send(comm, data, root);
    return {};
}

template <typename T>
void Gather_send(MPI_Comm comm, const std::vector<T>& data, int root) {
    MPI_Gather(data.data(), data.size(), types::get_mpi_type<T>(), nullptr, -1,
               MPI_DATATYPE_NULL, root, comm);
}

template <typename T>
std::vector<T> Gather_recv(MPI_Comm comm, const std::vector<T>& data) {
    std::vector<T> out(Comm_size(comm) * data.size());

    MPI_Gather(data.data(), data.size(), types::get_mpi_type<T>(), out.data(),
               data.size(), types::get_mpi_type<T>(), Comm_rank(comm), comm);
}

// ===================== Gatherv =====================
template <typename T>
void Gatherv_send(MPI_Comm comm, const std::vector<T>& data, int root) {
    int my_count = data.size();
    MPI_Gather(&my_count, 1, MPI_INT, nullptr, -1, MPI_DATATYPE_NULL, root,
               comm);

    MPI_Gatherv(data.data(), data.size(), types::get_mpi_type<T>(), nullptr,
                nullptr, nullptr, MPI_DATATYPE_NULL, root, comm);
}

template <typename T>
std::vector<std::vector<T>> Gatherv_recv(MPI_Comm comm,
                                         const std::vector<T>& data) {
    std::vector<T> counts(Comm_size(comm));
    int my_count = data.size();
    int my_rank = Comm_rank(comm);

    MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, my_rank, comm);

    std::vector<T> buffer(std::accumulate(counts.begin(), counts.end(), 0));
    std::vector<T> displs(counts.size());
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin());

    MPI_Gatherv(data.data(), data.size(), types::get_mpi_type<T>(),
                buffer.data(), counts.data(), displs.data(),
                types::get_mpi_type<T>(), my_rank, comm);

    std::vector<std::vector<T>> out(displs.size());
    for (std::size_t i = 0; i < displs.size(); ++i) {
        int start = displs[i];
        int end = i + 1 == displs.size() ? buffer.size() : displs[i + 1];

        out[i].insert(out[i].begin(), buffer.begin() + start,
                      buffer.begin() + end);
    }

    return out;
}

} // namespace MPIw