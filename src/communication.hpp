#pragma once

#include "getters.hpp"
#include "structs.hpp"
#include "types.hpp"
#include <mpi.h>
#include <numeric>
#include <algorithm>

namespace MPIw {

namespace details {
template <typename T>
std::vector<std::vector<T>> split_buffer(const std::vector<T>& buffer,
                                         const std::vector<T>& offsets) {
    std::vector<std::vector<T>> out(offsets.size());
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        int start = offsets[i];
        int end = (i + 1 == offsets.size()) ? buffer.size() : offsets[i + 1];

        out[i].insert(out[i].begin(), buffer.begin() + start,
                      buffer.begin() + end);
    }

    return out;
}
} // namespace details

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
    return data;
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
    if (Comm_rank(comm) == root)
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

    return out;
}

// ===================== Allgather =====================
template <typename T>
std::vector<T> Allgather(MPI_Comm comm, const std::vector<T> data) {
    std::vector<T> out(Comm_size(comm));

    MPI_Allgather(data.data(), data.size(), types::get_mpi_type<T>(),
                  out.data(), out.size(), types::get_mpi_type<T>(), comm);
    return out;
}

// ===================== Gatherv =====================
template <typename T>
std::vector<std::vector<T>>
Gatherv(MPI_Comm comm, const std::vector<T>& data, int root) {
    if (Comm_rank(comm) == root)
        return Gatherv_recv(comm, data);

    Gatherv_send(comm, data, root);
    return {};
}

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
    std::vector<int> counts(Comm_size(comm));
    int my_count = data.size();
    int my_rank = Comm_rank(comm);

    MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, my_rank, comm);

    std::vector<T> buffer(std::accumulate(counts.begin(), counts.end(), 0));
    std::vector<int> displs(counts.size());
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);

    MPI_Gatherv(data.data(), data.size(), types::get_mpi_type<T>(),
                buffer.data(), counts.data(), displs.data(),
                types::get_mpi_type<T>(), my_rank, comm);

    return details::split_buffer(buffer, displs);
}
// ===================== Allgatherv =====================
template <typename T>
std::vector<std::vector<T>> Allgatherv(MPI_Comm comm,
                                       const std::vector<T>& data) {
    int my_count = data.size();
    std::vector<int> counts(Comm_size(comm));

    MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(counts.size());
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
    std::vector<T> buffer(std::accumulate(counts.begin(), counts.end(), 0));

    MPI_Allgatherv(data.data(), data.size(), types::get_mpi_type<T>(),
                   buffer.data(), counts.data(), displs.data(),
                   types::get_mpi_type<T>(), comm);

    return details::split_buffer(buffer, displs);
}

// ===================== Scatter =====================
template <typename T>
std::vector<T>
Scatter(MPI_Comm comm, const std::vector<T>& data, int count, int root) {
    if (Comm_rank(comm) == root)
        return Scatter_send(comm, data);
    return Scatter_recv<T>(comm, count, root);
}

template <typename T>
std::vector<T> Scatter_send(MPI_Comm comm, const std::vector<T>& data) {
    int count = data.size() / Comm_size(comm);
    assert(Comm_size(comm) * count ==
           data.size()); // data are equally splitable

    std::vector<T> out(count);
    MPI_Scatter(data.begin(), count, types::get_mpi_type<T>(), out.data(),
                count, types::get_mpi_type<T>(), Comm_rank(comm), comm);
    return out;
}

template <typename T>
std::vector<T> Scatter_recv(MPI_Comm comm, int count, int root) {
    std::vector<T> out(count);
    MPI_Scatter(nullptr, -1, MPI_DATATYPE_NULL, out.data(), out.size(),
                types::get_mpi_type<T>(), root, comm);
    return out;
}

template <typename T>
std::vector<T> Scatter_send_managed(MPI_Comm comm, const std::vector<T>& data) {
    int count = data.size() / Comm_size(comm);
    MPI_Bcast(&count, 1, MPI_INT, Comm_rank(comm), comm);

    return Scatter_send(comm, data);
}

template <typename T>
std::vector<T> Scatter_recv_managed(MPI_Comm comm, int root) {
    int count;
    MPI_Bcast(&count, 1, MPI_INT, root, comm);
    return Scatter_recv<T>(comm, count, root);
}

// ===================== Scatterv =====================
template <typename T>
std::vector<T>
Scatterv(MPI_Comm comm, const std::vector<std::vector<T>>& data, int root) {
    if (Comm_rank(comm) == root)
        return Scatterv_send(comm, data);
    return Scatterv_recv<T>(comm, root);
}

template <typename T>
std::vector<T> Scatterv_send(MPI_Comm comm,
                             const std::vector<std::vector<T>>& data) {
    assert(data.size() == Comm_size(comm));

    std::vector<int> counts(data.size());
    std::vector<T> buffer;
    std::ranges::transform(data, counts.begin(), [&](const auto& v) {
        buffer.insert(buffer.end(), v.begin(),
                      v.end()); // filling buffer simultaneously
        return v.size();
    });
    std::vector<int> displs(counts.size());
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);

    int my_count;
    int my_rank = Comm_rank(comm);
    MPI_Scatter(counts.data(), 1, MPI_INT, &my_count, 1, MPI_INT, my_rank,
                comm);

    std::vector<T> out(my_count);
    MPI_Scatterv(buffer.data(), counts.data(), displs.data(),
                 types::get_mpi_type<T>(), out.data(), my_count,
                 types::get_mpi_type<T>(), my_rank, comm);

    return out;
}

template <typename T>
std::vector<T> Scatterv_recv(MPI_Comm comm, int root) {
    int my_count;
    MPI_Scatter(nullptr, -1, MPI_DATATYPE_NULL, &my_count, 1, MPI_INT, root,
                comm);

    std::vector<T> out(my_count);
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, out.data(),
                 out.size(), types::get_mpi_type<T>(), root, comm);

    return out;
}

// ===================== Reduce =====================
template <typename T>
std::vector<T>
Reduce(MPI_Comm comm, const std::vector<T>& data, MPI_Op op, int root) {
    if (Comm_rank(comm) == root)
        return Reduce_recv(comm, data, op);
    Reduce_send(comm, data, op, root);
    return {};
}

template <typename T>
void Reduce_send(MPI_Comm comm,
                 const std::vector<T>& data,
                 MPI_Op op,
                 int root) {
    MPI_Reduce(data.data(), nullptr, data.size(), types::get_mpi_type<T>(), op,
               root, comm);
}

template <typename T>
std::vector<T>
Reduce_recv(MPI_Comm comm, const std::vector<T>& data, MPI_Op op) {
    std::vector<T> out(data.size());

    MPI_Reduce(data.data(), out.data(), data.size(), types::get_mpi_type<T>(),
               op, Comm_rank(comm), comm);
    return out;
}

// ===================== AllReduce =====================
template <typename T>
std::vector<T> AllReduce(MPI_Comm comm, std::vector<T>& data, MPI_Op op) {
    std::vector<T> out(data.size());

    MPI_Allreduce(data.data(), out.data(), data.size(),
                  types::get_mpi_type<T>(), op, comm);
    return out;
}

// ===================== Barrier =====================
inline void Barrier(MPI_Comm comm) { MPI_Barrier(comm); }

} // namespace MPIw