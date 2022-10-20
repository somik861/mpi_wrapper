#pragma once

#include "error_codes.hpp"
#include "getters.hpp"
#include "structs.hpp"
#include "types.hpp"
#include <algorithm>
#include <cassert>
#include <mpi.h>
#include <numeric>
#include <source_location>

namespace MPIw {

namespace details {
template <typename T>
std::vector<std::vector<T>> split_buffer(const std::vector<T>& buffer,
                                         const std::vector<int>& offsets) {
	std::vector<std::vector<T>> out(offsets.size());
	for (std::size_t i = 0; i < offsets.size(); ++i) {
		int start = offsets[i];
		int end =
		    (i + 1 == offsets.size()) ? int(buffer.size()) : offsets[i + 1];

		out[i].insert(out[i].begin(), buffer.begin() + start,
		              buffer.begin() + end);
	}

	return out;
}
} // namespace details

// ===================== BASIC Send/Recv =====================

template <typename T>
structs::Recv_st<std::vector<T>>
Recv(MPI_Comm comm,
     int source = MPI_ANY_SOURCE,
     int tag = MPI_ANY_TAG,
     const std::source_location& location = std::source_location::current()) {
	structs::Recv_st<std::vector<T>> out;
	MPI_Status stat;
	MPI_Datatype type = types::get_mpi_type<T>();

	errors::check_code(MPI_Probe(source, tag, comm, &stat), location);

	out.data.resize(Get_count<T>(stat));
	errors::check_code(MPI_Recv(out.data.data(), int(out.data.size()), type,
	                            stat.MPI_SOURCE, stat.MPI_TAG, comm,
	                            &out.status),
	                   location);

	assert(stat.MPI_SOURCE == out.status.MPI_SOURCE);
	assert(stat.MPI_TAG == out.status.MPI_TAG);
	assert(int(out.data.size()) == Get_count<T>(out.status));

	return out;
}

template <typename T>
structs::Recv_st<T> Recv_one(
    MPI_Comm comm,
    int source = MPI_ANY_SOURCE,
    int tag = MPI_ANY_TAG,
    const std::source_location& location = std::source_location::current()) {
	structs::Recv_st<T> out;

	errors::check_code(MPI_Recv(&out.data, 1, types::get_mpi_type<T>(), source,
	                            tag, comm, &out.status),
	                   location);
	return out;
}

template <typename T>
void Send(
    MPI_Comm comm,
    const std::vector<T>& data,
    int dest,
    int tag,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Send(data.data(), int(data.size()),
	                            types::get_mpi_type<T>(), dest, tag, comm),
	                   location);
}

template <typename T>
void Send_one(
    MPI_Comm comm,
    T data,
    int dest,
    int tag,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(
	    MPI_Send(&data, 1, types::get_mpi_type<T>(), dest, tag, comm),
	    location);
}

// ===================== Bcast =====================

template <typename T>
std::vector<T>
Bcast(MPI_Comm comm,
      const std::vector<T>& data,
      int count,
      int root,
      const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) != root)
		return Bcast_recv<T>(comm, count, root, location);

	Bcast_send(comm, data, location);
	return data;
}

template <typename T>
std::vector<T> Bcast_managed(
    MPI_Comm comm,
    const std::vector<T>& data,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) != root)
		return Bcast_recv_managed<T>(comm, count, root, location);

	Bcast_send_managed(comm, data, location);
	return {};
}

template <typename T>
void Bcast_send(
    MPI_Comm comm,
    std::vector<T> data,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Bcast(data.data(), int(data.size()),
	                             types::get_mpi_type<T>(), Comm_rank(comm),
	                             comm),
	                   location);
}

template <typename T>
void Bcast_send_one(
    MPI_Comm comm,
    T data,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(
	    MPI_Bcast(&data, 1, types::get_mpi_type<T>(), Comm_rank(comm), comm),
	    location);
}

template <typename T>
std::vector<T> Bcast_recv(
    MPI_Comm comm,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(count);
	errors::check_code(
	    MPI_Bcast(out.data(), count, types::get_mpi_type<T>(), root, comm),
	    location);
	return out;
}

template <typename T>
T Bcast_recv_one(
    MPI_Comm comm,
    int root,
    const std::source_location& location = std::source_location::current()) {
	T out;
	errors::check_code(MPI_Bcast(&out, 1, types::get_mpi_type<T>(), root, comm),
	                   location);
	return out;
}

template <typename T>
void Bcast_send_managed(
    MPI_Comm comm,
    const std::vector<T>& data,
    const std::source_location& location = std::source_location::current()) {
	int count = int(data.size());
	int my_rank = Comm_rank(comm);
	errors::check_code(MPI_Bcast(&count, 1, MPI_INT, my_rank, comm), location);
	Bcast_send(comm, data, location);
}

template <typename T>
std::vector<T> Bcast_recv_managed(
    MPI_Comm comm,
    int root,
    const std::source_location& location = std::source_location::current()) {
	int count;
	errors::check_code(MPI_Bcast(&count, 1, MPI_INT, root, comm), location);
	return Bcast_recv<T>(comm, count, root, location);
}

// ===================== Gather =====================
template <typename T>
std::vector<T>
Gather(MPI_Comm comm,
       const std::vector<T>& data,
       int root,
       const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) == root)
		return Gather_recv(comm, data, location);

	Gather_send(comm, data, root, location);
	return {};
}

template <typename T>
void Gather_send(
    MPI_Comm comm,
    const std::vector<T>& data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Gather(data.data(), int(data.size()),
	                              types::get_mpi_type<T>(), nullptr, -1,
	                              MPI_DATATYPE_NULL, root, comm),
	                   location);
}

template <typename T>
void Gather_send_one(
    MPI_Comm comm,
    T data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Gather(&data, 1, types::get_mpi_type<T>(), nullptr,
	                              -1, MPI_DATATYPE_NULL, root, comm),
	                   location);
}

template <typename T>
std::vector<T> Gather_recv(
    MPI_Comm comm,
    const std::vector<T>& data,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(Comm_size(comm) * data.size());

	errors::check_code(MPI_Gather(data.data(), int(data.size()),
	                              types::get_mpi_type<T>(), out.data(),
	                              int(data.size()), types::get_mpi_type<T>(),
	                              Comm_rank(comm), comm),
	                   location);

	return out;
}

template <typename T>
std::vector<T> Gather_recv_one(
    MPI_Comm comm,
    T data,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(Comm_size(comm));

	errors::check_code(MPI_Gather(&data, 1, types::get_mpi_type<T>(),
	                              out.data(), 1, types::get_mpi_type<T>(),
	                              Comm_rank(comm), comm),
	                   location);

	return out;
}

// ===================== Allgather =====================
template <typename T>
std::vector<T> Allgather(
    MPI_Comm comm,
    const std::vector<T> data,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(Comm_size(comm) * data.size());

	errors::check_code(MPI_Allgather(data.data(), int(data.size()),
	                                 types::get_mpi_type<T>(), out.data(),
	                                 int(data.size()), types::get_mpi_type<T>(),
	                                 comm),
	                   location);
	return out;
}

// ===================== Gatherv =====================
template <typename T>
std::vector<std::vector<T>> Gatherv(
    MPI_Comm comm,
    const std::vector<T>& data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) == root)
		return Gatherv_recv(comm, data, location);

	Gatherv_send(comm, data, root, location);
	return {};
}

template <typename T>
void Gatherv_send(
    MPI_Comm comm,
    const std::vector<T>& data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	int my_count = int(data.size());
	errors::check_code(MPI_Gather(&my_count, 1, MPI_INT, nullptr, -1,
	                              MPI_DATATYPE_NULL, root, comm),
	                   location);

	errors::check_code(MPI_Gatherv(data.data(), int(data.size()),
	                               types::get_mpi_type<T>(), nullptr, nullptr,
	                               nullptr, MPI_DATATYPE_NULL, root, comm),
	                   location);
}

template <typename T>
std::vector<std::vector<T>> Gatherv_recv(
    MPI_Comm comm,
    const std::vector<T>& data,
    const std::source_location& location = std::source_location::current()) {
	std::vector<int> counts(Comm_size(comm));
	int my_count = int(data.size());
	int my_rank = Comm_rank(comm);

	errors::check_code(MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1,
	                              MPI_INT, my_rank, comm),
	                   location);

	std::vector<T> buffer(std::accumulate(counts.begin(), counts.end(), 0));
	std::vector<int> displs(counts.size());
	std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);

	errors::check_code(MPI_Gatherv(data.data(), int(data.size()),
	                               types::get_mpi_type<T>(), buffer.data(),
	                               counts.data(), displs.data(),
	                               types::get_mpi_type<T>(), my_rank, comm),
	                   location);

	return details::split_buffer(buffer, displs);
}
// ===================== Allgatherv =====================
template <typename T>
std::vector<std::vector<T>> Allgatherv(
    MPI_Comm comm,
    const std::vector<T>& data,
    const std::source_location& location = std::source_location::current()) {
	int my_count = int(data.size());
	std::vector<int> counts(Comm_size(comm));

	errors::check_code(
	    MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm),
	    location);

	std::vector<int> displs(counts.size());
	std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
	std::vector<T> buffer(std::accumulate(counts.begin(), counts.end(), 0));

	errors::check_code(MPI_Allgatherv(data.data(), int(data.size()),
	                                  types::get_mpi_type<T>(), buffer.data(),
	                                  counts.data(), displs.data(),
	                                  types::get_mpi_type<T>(), comm),
	                   location);

	return details::split_buffer(buffer, displs);
}

// ===================== Scatter =====================
template <typename T>
std::vector<T> Scatter(
    MPI_Comm comm,
    const std::vector<T>& data,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) == root)
		return Scatter_send(comm, data, location);
	return Scatter_recv<T>(comm, count, root, location);
}

template <typename T>
std::vector<T> Scatter_send(
    MPI_Comm comm,
    const std::vector<T>& data,
    const std::source_location& location = std::source_location::current()) {
	int count = int(data.size()) / Comm_size(comm);
	assert(Comm_size(comm) * count ==
	       int(data.size())); // data are equally splitable

	std::vector<T> out(count);
	errors::check_code(MPI_Scatter(data.data(), count, types::get_mpi_type<T>(),
	                               out.data(), count, types::get_mpi_type<T>(),
	                               Comm_rank(comm), comm),
	                   location);
	return out;
}

template <typename T>
std::vector<T> Scatter_recv(
    MPI_Comm comm,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(count);
	errors::check_code(MPI_Scatter(nullptr, -1, MPI_DATATYPE_NULL, out.data(),
	                               int(out.size()), types::get_mpi_type<T>(),
	                               root, comm),
	                   location);
	return out;
}

template <typename T>
std::vector<T> Scatter_send_managed(
    MPI_Comm comm,
    const std::vector<T>& data,
    const std::source_location& location = std::source_location::current()) {
	int count = int(data.size()) / Comm_size(comm);
	errors::check_code(MPI_Bcast(&count, 1, MPI_INT, Comm_rank(comm), comm),
	                   location);

	return Scatter_send(comm, data, location);
}

template <typename T>
std::vector<T> Scatter_recv_managed(
    MPI_Comm comm,
    int root,
    const std::source_location& location = std::source_location::current()) {
	int count;
	errors::check_code(MPI_Bcast(&count, 1, MPI_INT, root, comm), location);
	return Scatter_recv<T>(comm, count, root, location);
}

// ===================== Scatterv =====================
template <typename T>
std::vector<T> Scatterv(
    MPI_Comm comm,
    const std::vector<std::vector<T>>& data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) == root)
		return Scatterv_send(comm, data, location);
	return Scatterv_recv<T>(comm, root, location);
}

template <typename T>
std::vector<T> Scatterv_send(
    MPI_Comm comm,
    const std::vector<std::vector<T>>& data,
    const std::source_location& location = std::source_location::current()) {
	assert(int(data.size()) == Comm_size(comm));

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
	errors::check_code(MPI_Scatter(counts.data(), 1, MPI_INT, &my_count, 1,
	                               MPI_INT, my_rank, comm),
	                   location);

	std::vector<T> out(my_count);
	errors::check_code(MPI_Scatterv(buffer.data(), counts.data(), displs.data(),
	                                types::get_mpi_type<T>(), out.data(),
	                                my_count, types::get_mpi_type<T>(), my_rank,
	                                comm),
	                   location);

	return out;
}

template <typename T>
std::vector<T> Scatterv_recv(
    MPI_Comm comm,
    int root,
    const std::source_location& location = std::source_location::current()) {
	int my_count;
	errors::check_code(MPI_Scatter(nullptr, -1, MPI_DATATYPE_NULL, &my_count, 1,
	                               MPI_INT, root, comm),
	                   location);

	std::vector<T> out(my_count);
	errors::check_code(
	    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, out.data(),
	                 int(out.size()), types::get_mpi_type<T>(), root, comm),
	    location);

	return out;
}

// ===================== Reduce =====================
template <typename T>
std::vector<T>
Reduce(MPI_Comm comm,
       const std::vector<T>& data,
       MPI_Op op,
       int root,
       const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) == root)
		return Reduce_recv(comm, data, op, location);
	Reduce_send(comm, data, op, root, location);
	return {};
}

template <typename T>
void Reduce_send(
    MPI_Comm comm,
    const std::vector<T>& data,
    MPI_Op op,
    int root,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Reduce(data.data(), nullptr, int(data.size()),
	                              types::get_mpi_type<T>(), op, root, comm),
	                   location);
}

template <typename T>
std::vector<T> Reduce_recv(
    MPI_Comm comm,
    const std::vector<T>& data,
    MPI_Op op,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(data.size());

	errors::check_code(MPI_Reduce(data.data(), out.data(), int(data.size()),
	                              types::get_mpi_type<T>(), op, Comm_rank(comm),
	                              comm),
	                   location);
	return out;
}

// ===================== AllReduce =====================
template <typename T>
std::vector<T> AllReduce(
    MPI_Comm comm,
    const std::vector<T>& data,
    MPI_Op op,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(data.size());

	errors::check_code(MPI_Allreduce(data.data(), out.data(), int(data.size()),
	                                 types::get_mpi_type<T>(), op, comm),
	                   location);
	return out;
}

// ===================== Barrier =====================
inline void Barrier(
    MPI_Comm comm,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Barrier(comm), location);
}

} // namespace MPIw