#pragma once

#include "../src/include.hpp"
#include "utils.hpp"
#include <fmt/core.h>
#include <numeric>
#include <random>

namespace master {
template <typename T>
void run_broadcast_one(MPI_Comm comm) {
    T to_send = T(-1);
    MPIw::Bcast_send_one(comm, to_send);
}

template <typename T>
void run_broadcast_single(MPI_Comm comm) {
    std::vector<T> to_send = {T(-1)};
    MPIw::Bcast_send(comm, to_send);
}

template <typename T>
void run_broadcast_range(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    std::vector<T> to_send(comm_size);
    std::iota(to_send.begin(), to_send.end(), my_rank);

    MPIw::Bcast_send(comm, to_send);
}

template <typename T>
void run_broadcast_managed(MPI_Comm comm) {
    for (int n = 0; n < 10; ++n) {
        int size = std::random_device{}() % 100 + 1;
        std::vector<T> to_send(size);
        std::iota(to_send.begin(), to_send.end(), size);

        MPIw::Bcast_send_managed(comm, to_send);
    }
}
} // namespace master

namespace slave {
template <typename T>
void run_broadcast_one(MPI_Comm comm, int root) {
    T expected = T(-1);

    auto data = MPIw::Bcast_recv_one<T>(comm, root);
    assert(data == expected);
}

template <typename T>
void run_broadcast_single(MPI_Comm comm, int root) {
    std::vector<T> expected = {T(-1)};

    auto data = MPIw::Bcast_recv<T>(comm, 1, root);
    assert(data == expected);
}

template <typename T>
void run_broadcast_range(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    std::vector<T> expected(comm_size);
    std::iota(expected.begin(), expected.end(), root);

    auto data = MPIw::Bcast_recv<T>(comm, comm_size, root);
    assert(data == expected);
}

template <typename T>
void run_broadcast_managed(MPI_Comm comm, int root) {
    for (int n = 0; n < 10; ++n) {
        auto data = MPIw::Bcast_recv_managed<T>(comm, root);
        assert(data.size() == std::size_t(data[0]));
        std::vector<T> expected(data.size());
        std::iota(expected.begin(), expected.end(), data.size());
        assert(data == expected);
    }
}
} // namespace slave

template <typename T>
void run_broadcast(MPI_Comm comm) {
    int rank = MPIw::Comm_rank(comm);
    int size = MPIw::Comm_size(comm);
    bool is_printing = (rank == 0);

    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(fmt::format("broadcast one (root: {})", root));
        if (root != rank)
            slave::run_broadcast_one<T>(comm, root);
        else {
            master::run_broadcast_one<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("broadcast one");
    }

    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("broadcast single (root: {})", root));
        if (root != rank)
            slave::run_broadcast_single<T>(comm, root);
        else {
            master::run_broadcast_single<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("broadcast single");
    }

    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("broadcast range (root: {})", root));
        if (root != rank)
            slave::run_broadcast_range<T>(comm, root);
        else {
            master::run_broadcast_range<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("broadcast range");
    }

    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("broadcast managed (root: {})", root));
        if (root != rank)
            slave::run_broadcast_managed<T>(comm, root);
        else {
            master::run_broadcast_managed<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("broadcast managed");
    }
}