#pragma once

#include "../src/include.hpp"
#include "utils.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace master {
template <typename T>
void run_reduce_single(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    auto sum = MPIw::Reduce_recv<T>(comm, {T(my_rank)}, MPI_SUM);
    assert(sum.size() == 1);

    T expected = T((comm_size * (comm_size - 1)) / 2);
    assert(sum[0] == expected);
}

template <typename T>
void run_reduce_range(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    std::vector<T> to_send(comm_size, T(my_rank));
    for (std::size_t i = 0; i < to_send.size(); ++i)
        to_send[i] *= T(i);

    auto got = MPIw::Reduce_recv<T>(comm, to_send, MPI_SUM);
    assert(got.size() == std::size_t(comm_size));

    T sum = T((comm_size * (comm_size - 1)) / 2);
    for (std::size_t i = 0; i < got.size(); ++i)
        assert(got[i] == sum * T(i));
}

template <typename T>
void run_allreduce_single(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    auto sum = MPIw::AllReduce<T>(comm, {T(my_rank)}, MPI_SUM);
    assert(sum.size() == 1);

    T expected = T((comm_size * (comm_size - 1)) / 2);
    assert(sum[0] == expected);
}

template <typename T>
void run_allreduce_range(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    std::vector<T> to_send(comm_size, T(my_rank));
    for (std::size_t i = 0; i < to_send.size(); ++i)
        to_send[i] *= T(i);

    auto got = MPIw::AllReduce<T>(comm, to_send, MPI_SUM);
    assert(got.size() == std::size_t(comm_size));

    T sum = T((comm_size * (comm_size - 1)) / 2);
    for (std::size_t i = 0; i < got.size(); ++i)
        assert(got[i] == sum * T(i));
}

} // namespace master

namespace slave {
template <typename T>
void run_reduce_single(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);
    MPIw::Reduce_send<T>(comm, {T(my_rank)}, MPI_SUM, root);
}

template <typename T>
void run_reduce_range(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    std::vector<T> to_send(comm_size, T(my_rank));
    for (std::size_t i = 0; i < to_send.size(); ++i)
        to_send[i] *= T(i);

    MPIw::Reduce_send<T>(comm, to_send, MPI_SUM, root);
}
} // namespace slave

template <typename T>
void run_reduce(MPI_Comm comm) {
    int rank = MPIw::Comm_rank(comm);
    int size = MPIw::Comm_size(comm);
    bool is_printing = (rank == 0);

    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(fmt::format("reduce single (root: {})", root));
        if (root != rank)
            slave::run_reduce_single<T>(comm, root);
        else {
            master::run_reduce_single<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("reduce single");
    }

    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(fmt::format("reduce range (root: {})", root));
        if (root != rank)
            slave::run_reduce_range<T>(comm, root);
        else {
            master::run_reduce_range<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("reduce range");
    }

    if (is_printing)
        print_start_section("allreduce single");
    master::run_allreduce_single<T>(comm);
    MPIw::Barrier(comm);
    if (is_printing)
        print_end_section("allreduce single");

    if (is_printing)
        print_start_section("allreduce range");
    master::run_allreduce_range<T>(comm);
    MPIw::Barrier(comm);
    if (is_printing)
        print_end_section("allreduce range");
}