#pragma once

#include "../src/include.hpp"
#include "utils.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace master {
template <typename T>
void run_gather_scather_single(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    std::vector<T> to_send(comm_size);
    std::iota(to_send.begin(), to_send.end(), 0);
    std::ranges::transform(to_send, to_send.begin(),
                           [](auto x) { return 2 * x; });

    auto part = MPIw::Scatter_send(comm, to_send);
    assert(part.size() == 1);
    assert(part[0] == my_rank * 2);

    auto full = MPIw::Gather_recv(comm, part);
    assert(full == to_send);
}

template <typename T>
void run_gather_scather_managed(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    for (int n = 0; n < 10; ++n) {
        int size = std::random_device{}() % 100;

        std::vector<T> to_send(comm_size * size);
        std::iota(to_send.begin(), to_send.end(), 0);
        std::ranges::transform(to_send, to_send.begin(),
                               [](auto x) { return 2 * x; });

        auto part = MPIw::Scatter_send_managed(comm, to_send);
        auto full = MPIw::Gather_recv(comm, part);
        assert(full == to_send);
    }
}

template <typename T>
void run_gather_scather_variable(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    for (int n = 0; n < 10; ++n) {
        std::vector<std::vector<T>> to_send;
        for (int i = 0; i < comm_size; ++i) {
            int size = std::random_device{}() % 100;
            std::vector<T> tmp(size);
            std::iota(tmp.begin(), tmp.end(), 0);
            to_send.push_back(std::move(tmp));
        }
        auto part = MPIw::Scatterv_send(comm, to_send);
        auto full = MPIw::Gatherv_recv(comm, part);
        assert(full == to_send);
    }
}

} // namespace master

namespace slave {
template <typename T>
void run_gather_scather_single(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);

    auto part = MPIw::Scatter_recv<T>(comm, 1, root);
    assert(part.size() == 1);
    assert(part[0] == my_rank * 2);

    MPIw::Gather_send(comm, part, root);
}

template <typename T>
void run_gather_scather_managed(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);

    for (int n = 0; n < 10; ++n) {
        auto part = MPIw::Scatter_recv_managed<T>(comm, root);
        MPIw::Gather_send(comm, part, root);
    }
}

template <typename T>
void run_gather_scather_variable(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);

    for (int n = 0; n < 10; ++n) {
        auto part = MPIw::Scatterv_recv<T>(comm, root);
        MPIw::Gatherv_send(comm, part, root);
    }
}

} // namespace slave

template <typename T>
void run_gather_scather(MPI_Comm comm) {
    int rank = MPIw::Comm_rank(comm);
    int size = MPIw::Comm_size(comm);
    bool is_printing = (rank == 0);

    // single
    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("gather / scather single (root: {})", root));
        if (root != rank)
            slave::run_gather_scather_single<T>(comm, root);
        else {
            master::run_gather_scather_single<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("gather / scather single");
    }

    // managed
    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("gather / scather managed (root: {})", root));
        if (root != rank)
            slave::run_gather_scather_managed<T>(comm, root);
        else {
            master::run_gather_scather_managed<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("gather / scather managed");
    }

    // variable
    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("gather / scather variable (root: {})", root));
        if (root != rank)
            slave::run_gather_scather_variable<T>(comm, root);
        else {
            master::run_gather_scather_variable<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("gather / scather variable");
    }
}