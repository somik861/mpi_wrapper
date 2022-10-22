#pragma once

#include "concepts.hpp"
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

template <typename T, details::cnpts::EnumOrInt U = int>
MPI_Status
Recv(MPI_Comm comm,
     T* dest,
     int count,
     int source = MPI_ANY_SOURCE,
     U tag = MPI_ANY_TAG,
     const std::source_location& location = std::source_location::current()) {
	MPI_Status stat;

	errors::check_code(MPI_Recv(dest, count, types::get_mpi_type<T>(), source,
	                            tag, comm, &stat),
	                   location);

	return stat;
}

template <typename T, details::cnpts::EnumOrInt U = int>
structs::Recv_st<std::vector<T>>
Recv(MPI_Comm comm,
     int source = MPI_ANY_SOURCE,
     U tag = MPI_ANY_TAG,
     const std::source_location& location = std::source_location::current()) {
	structs::Recv_st<std::vector<T>> out;

	MPI_Status stat;
	errors::check_code(MPI_Probe(source, static_cast<int>(tag), comm, &stat),
	                   location);

	int count = Get_count<T>(stat);
	out.data.resize(count);
	out.status = Recv(comm, out.data.data(), count, stat.MPI_SOURCE,
	                  stat.MPI_TAG, location);

	assert(stat.MPI_SOURCE == out.status.MPI_SOURCE);
	assert(stat.MPI_TAG == out.status.MPI_TAG);
	assert(int(out.data.size()) == Get_count<T>(out.status));

	return out;
}

template <typename T, details::cnpts::EnumOrInt U = int>
structs::Recv_st<T> Recv_one(
    MPI_Comm comm,
    int source = MPI_ANY_SOURCE,
    U tag = MPI_ANY_TAG,
    const std::source_location& location = std::source_location::current()) {
	structs::Recv_st<T> out;

	out.status = Recv(comm, &out.data, 1, source, tag, location);
	return out;
}

template <typename T, details::cnpts::EnumOrInt U = int>
void Send(
    MPI_Comm comm,
    const T* data,
    int count,
    int dest,
    U tag,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Send(data, count, types::get_mpi_type<T>(), dest,
	                            static_cast<int>(tag), comm),
	                   location);
}

template <details::cnpts::Container T, details::cnpts::EnumOrInt U = int>
void Send(
    MPI_Comm comm,
    const T& data,
    int dest,
    U tag,
    const std::source_location& location = std::source_location::current()) {
	Send(comm, &*data.begin(), int(data.size()), dest, tag, location);
}

template <typename T, details::cnpts::EnumOrInt U = int>
void Send_one(
    MPI_Comm comm,
    T data,
    int dest,
    U tag,
    const std::source_location& location = std::source_location::current()) {
	Send(comm, &data, 1, dest, tag, location);
}

// ===================== Bcast =====================

template <details::cnpts::Container T>
std::vector<typename T::value_type>
Bcast(MPI_Comm comm,
      const T& data,
      int count,
      int root,
      const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) != root)
		return Bcast_recv<T>(comm, count, root, location);

	Bcast_send(comm, data, location);
	return data;
}

template <details::cnpts::Container T>
std::vector<typename T::value_type> Bcast_managed(
    MPI_Comm comm,
    const T& data,
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
    const T* data,
    int count,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Bcast(const_cast<T*>(data), count,
	                             types::get_mpi_type<T>(), Comm_rank(comm),
	                             comm),
	                   location);
}

template <details::cnpts::Container T>
void Bcast_send(
    MPI_Comm comm,
    const T& data,
    const std::source_location& location = std::source_location::current()) {
	Bcast_send(comm, &*data.begin(), int(data.size()), location);
}

template <typename T>
void Bcast_send_one(
    MPI_Comm comm,
    T data,
    const std::source_location& location = std::source_location::current()) {
	Bcast_send(comm, &data, 1, location);
}

template <typename T>
void Bcast_recv(
    MPI_Comm comm,
    T* dest,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(
	    MPI_Bcast(dest, count, types::get_mpi_type<T>(), root, comm), location);
}

template <typename T>
std::vector<T> Bcast_recv(
    MPI_Comm comm,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(count);
	Bcast_recv(comm, out.data(), count, root, location);
	return out;
}

template <typename T>
T Bcast_recv_one(
    MPI_Comm comm,
    int root,
    const std::source_location& location = std::source_location::current()) {
	T out;
	Bcast_recv(comm, &out, 1, root, location);
	return out;
}

template <typename T>
void Bcast_send_managed(
    MPI_Comm comm,
    const T* data,
    int count,
    const std::source_location& location = std::source_location::current()) {
	Bcast_send_one(comm, count, location);
	Bcast_send(comm, data, count, location);
}

template <details::cnpts::Container T>
void Bcast_send_managed(
    MPI_Comm comm,
    const T& data,
    const std::source_location& location = std::source_location::current()) {
	Bcast_send_managed(comm, &*data.begin(), int(data.size()), location);
}

template <typename T>
std::vector<T> Bcast_recv_managed(
    MPI_Comm comm,
    int root,
    const std::source_location& location = std::source_location::current()) {
	int count = Bcast_recv_one<int>(comm, root, location);
	return Bcast_recv<T>(comm, count, root, location);
}

// ===================== Gather =====================
template <details::cnpts::Container T>
std::vector<typename T::value_type>
Gather(MPI_Comm comm,
       const T& data,
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
    const T* data,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Gather(data, count, types::get_mpi_type<T>(),
	                              nullptr, -1, MPI_DATATYPE_NULL, root, comm),
	                   location);
}

template <details::cnpts::Container T>
void Gather_send(
    MPI_Comm comm,
    const T& data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	Gather_send(comm, &*data.begin(), int(data.size()), root, location);
}

template <typename T>
void Gather_send_one(
    MPI_Comm comm,
    T data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	Gather_send(comm, &data, 1, root, location);
}

template <typename T>
void Gather_recv(
    MPI_Comm comm,
    const T* data,
    T* dest,
    int count,
    const std::source_location& location = std::source_location::current()) {

	errors::check_code(MPI_Gather(data, count, types::get_mpi_type<T>(), dest,
	                              count, types::get_mpi_type<T>(),
	                              Comm_rank(comm), comm),
	                   location);
}

template <details::cnpts::Container T>
std::vector<typename T::value_type> Gather_recv(
    MPI_Comm comm,
    const T& data,
    const std::source_location& location = std::source_location::current()) {
	std::vector<typename T::value_type> out(Comm_size(comm) * data.size());

	Gather_recv(comm, &*data.begin(), out.data(), int(data.size()), location);
	return out;
}

template <typename T>
std::vector<T> Gather_recv_one(
    MPI_Comm comm,
    T data,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(Comm_size(comm));

	Gather_recv(comm, &data, out.data(), 1, location);
	return out;
}

// ===================== Allgather =====================
template <typename T>
void Allgather(
    MPI_Comm comm,
    const T* data,
    T* dest,
    int count,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Allgather(data, count, types::get_mpi_type<T>(),
	                                 dest, count, types::get_mpi_type<T>(),
	                                 comm),
	                   location);
}

template <details::cnpts::Container T>
std::vector<typename T::value_type> Allgather(
    MPI_Comm comm,
    const T& data,
    const std::source_location& location = std::source_location::current()) {
	std::vector<typename T::value_type> out(Comm_size(comm) * data.size());

	Allgather(comm, &*data.begin(), out.data(), int(data.size()), location);
	return out;
}

// ===================== Gatherv =====================
template <details::cnpts::Container T>
std::vector<std::vector<typename T::value_type>> Gatherv(
    MPI_Comm comm,
    const T& data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) == root)
		return Gatherv_recv(comm, data, location);

	Gatherv_send(comm, data, root, location);
	return {};
}

template <details::cnpts::Container T>
void Gatherv_send(
    MPI_Comm comm,
    const T& data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	Gather_send_one(comm, int(data.size()), root, location);

	errors::check_code(
	    MPI_Gatherv(&*data.begin(), int(data.size()),
	                types::get_mpi_type<typename T::value_type>(), nullptr,
	                nullptr, nullptr, MPI_DATATYPE_NULL, root, comm),
	    location);
}

template <details::cnpts::Container T>
std::vector<std::vector<typename T::value_type>> Gatherv_recv(
    MPI_Comm comm,
    const T& data,
    const std::source_location& location = std::source_location::current()) {
	int my_rank = Comm_rank(comm);
	using value_type = typename T::value_type;

	auto counts = Gather_recv_one(comm, int(data.size()), location);

	std::vector<value_type> buffer(
	    std::accumulate(counts.begin(), counts.end(), 0));
	std::vector<int> displs(counts.size());
	std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);

	errors::check_code(MPI_Gatherv(&*data.begin(), int(data.size()),
	                               types::get_mpi_type<value_type>(),
	                               buffer.data(), counts.data(), displs.data(),
	                               types::get_mpi_type<value_type>(), my_rank,
	                               comm),
	                   location);

	return details::split_buffer(buffer, displs);
}
// ===================== Allgatherv =====================
template <details::cnpts::Container T>
std::vector<std::vector<typename T::value_type>> Allgatherv(
    MPI_Comm comm,
    const T& data,
    const std::source_location& location = std::source_location::current()) {
	int my_count = int(data.size());
	std::vector<int> counts(Comm_size(comm));

	using value_type = typename T::value_type;

	Allgather(comm, &my_count, counts.data(), 1, location);

	std::vector<int> displs(counts.size());
	std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
	std::vector<value_type> buffer(
	    std::accumulate(counts.begin(), counts.end(), 0));

	errors::check_code(MPI_Allgatherv(&*data.begin(), int(data.size()),
	                                  types::get_mpi_type<value_type>(),
	                                  buffer.data(), counts.data(),
	                                  displs.data(),
	                                  types::get_mpi_type<value_type>(), comm),
	                   location);

	return details::split_buffer(buffer, displs);
}

// ===================== Scatter =====================
template <details::cnpts::Container T>
std::vector<typename T::value_type> Scatter(
    MPI_Comm comm,
    const T& data,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) == root)
		return Scatter_send(comm, data, location);
	return Scatter_recv<T>(comm, count, root, location);
}

template <typename T>
void Scatter_send(
    MPI_Comm comm,
    const T* data,
    T* dest,
    int total_count,
    const std::source_location& location = std::source_location::current()) {
	int count = total_count / Comm_size(comm);
	assert(Comm_size(comm) * count ==
	       total_count); // data are equally splitable

	errors::check_code(MPI_Scatter(data, count, types::get_mpi_type<T>(), dest,
	                               count, types::get_mpi_type<T>(),
	                               Comm_rank(comm), comm),
	                   location);
}

template <details::cnpts::Container T>
std::vector<typename T::value_type> Scatter_send(
    MPI_Comm comm,
    const T& data,
    const std::source_location& location = std::source_location::current()) {
	int count = int(data.size()) / Comm_size(comm);
	assert(Comm_size(comm) * count ==
	       int(data.size())); // data are equally splitable

	std::vector<typename T::value_type> out(count);
	Scatter_send(comm, &*data.begin(), out.data(), int(data.size()), location);
	return out;
}

template <typename T>
void Scatter_recv(
    MPI_Comm comm,
    T* dest,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Scatter(nullptr, -1, MPI_DATATYPE_NULL, dest, count,
	                               types::get_mpi_type<T>(), root, comm),
	                   location);
}

template <typename T>
std::vector<T> Scatter_recv(
    MPI_Comm comm,
    int count,
    int root,
    const std::source_location& location = std::source_location::current()) {
	std::vector<T> out(count);
	Scatter_recv(comm, out.data(), count, root, location);
	return out;
}

template <typename T>
void Scatter_send_managed(
    MPI_Comm comm,
    const T* data,
    T* dest,
    int total_count,
    const std::source_location& location = std::source_location::current()) {
	int count = total_count / Comm_size(comm);

	Bcast_send_one(comm, count, location);
	Scatter_send(comm, data, dest, total_count, location);
}

template <details::cnpts::Container T>
std::vector<typename T::value_type> Scatter_send_managed(
    MPI_Comm comm,
    const T& data,
    const std::source_location& location = std::source_location::current()) {
	int count = int(data.size()) / Comm_size(comm);
	std::vector<typename T::value_type> out(count);

	Scatter_send_managed(comm, &*data.begin(), out.data(), int(data.size()),
	                     location);
	return out;
}

template <typename T>
void Scatter_recv_managed(
    MPI_Comm comm,
    T* dest,
    int root,
    const std::source_location& location = std::source_location::current()) {
	int count = Bcast_recv_one<int>(comm, root, location);
	Scatter_recv(comm, dest, count, root, location);
}

template <typename T>
std::vector<T> Scatter_recv_managed(
    MPI_Comm comm,
    int root,
    const std::source_location& location = std::source_location::current()) {
	int count = Bcast_recv_one<int>(comm, root, location);
	return Scatter_recv<T>(comm, count, root, location);
}

// ===================== Scatterv =====================
template <details::cnpts::Container T>
std::vector<typename T::value_type> Scatterv(
    MPI_Comm comm,
    const std::vector<T>& data,
    int root,
    const std::source_location& location = std::source_location::current()) {
	if (Comm_rank(comm) == root)
		return Scatterv_send(comm, data, location);
	return Scatterv_recv<T>(comm, root, location);
}

template <details::cnpts::Container T>
std::vector<typename T::value_type> Scatterv_send(
    MPI_Comm comm,
    const std::vector<T>& data,
    const std::source_location& location = std::source_location::current()) {
	assert(int(data.size()) == Comm_size(comm));
	using U = typename T::value_type;

	std::vector<int> counts(data.size());
	std::vector<U> buffer;
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

	std::vector<U> out(my_count);
	errors::check_code(MPI_Scatterv(buffer.data(), counts.data(), displs.data(),
	                                types::get_mpi_type<U>(), out.data(),
	                                my_count, types::get_mpi_type<U>(), my_rank,
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
	Scatter_recv(comm, &my_count, 1, root, location);

	std::vector<T> out(my_count);
	errors::check_code(
	    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, out.data(),
	                 int(out.size()), types::get_mpi_type<T>(), root, comm),
	    location);

	return out;
}

// ===================== Reduce =====================
template <details::cnpts::Container T>
std::vector<typename T::value_type>
Reduce(MPI_Comm comm,
       const T& data,
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
    const T* data,
    int count,
    MPI_Op op,
    int root,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Reduce(data, nullptr, count,
	                              types::get_mpi_type<T>(), op, root, comm),
	                   location);
}

template <details::cnpts::Container T>
void Reduce_send(
    MPI_Comm comm,
    const T& data,
    MPI_Op op,
    int root,
    const std::source_location& location = std::source_location::current()) {
	Reduce_send(comm, &*data.begin(), int(data.size()), op, root, location);
}

template <typename T>
void Reduce_recv(
    MPI_Comm comm,
    const T* data,
    T* dest,
    int count,
    MPI_Op op,
    const std::source_location& location = std::source_location::current()) {

	errors::check_code(MPI_Reduce(data, dest, count, types::get_mpi_type<T>(),
	                              op, Comm_rank(comm), comm),
	                   location);
}

template <details::cnpts::Container T>
std::vector<typename T::value_type> Reduce_recv(
    MPI_Comm comm,
    const T& data,
    MPI_Op op,
    const std::source_location& location = std::source_location::current()) {
	std::vector<typename T::value_type> out(data.size());

	Reduce_recv(comm, &*data.begin(), out.data(), int(data.size()), op,
	            location);
	return out;
}

// ===================== AllReduce =====================
template <typename T>
void AllReduce(
    MPI_Comm comm,
    const T* data,
    T* dest,
    int count,
    MPI_Op op,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(
	    MPI_Allreduce(data, dest, count, types::get_mpi_type<T>(), op, comm),
	    location);
}

template <details::cnpts::Container T>
std::vector<typename T::value_type> AllReduce(
    MPI_Comm comm,
    const T& data,
    MPI_Op op,
    const std::source_location& location = std::source_location::current()) {
	std::vector<typename T::value_type> out(data.size());

	AllReduce(comm, &*data.begin(), out.data(), int(data.size()), op, location);

	return out;
}

// ===================== Barrier =====================
inline void Barrier(
    MPI_Comm comm,
    const std::source_location& location = std::source_location::current()) {
	errors::check_code(MPI_Barrier(comm), location);
}

} // namespace MPIw