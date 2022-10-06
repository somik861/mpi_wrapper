#pragma once

#include "../src/include.hpp"
#include "utils.hpp"
#include <set>

namespace master {
template <typename T>
void run_send_recv_one(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    for (int dest = 0; dest < comm_size; ++dest) {
        if (dest == my_rank)
            continue;

        MPIw::Send_one<T>(comm, T(dest * 3), dest, dest * 2);
        const auto [data, status] = MPIw::Recv_one<T>(comm);
        assert(status.MPI_SOURCE == dest);
        assert(status.MPI_TAG == dest * 5);
        assert(MPIw::Get_count<T>(status) == 1);
        assert(data == T(dest * 4));
    }
}

template <typename T>
void run_send_recv_single(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    for (int dest = 0; dest < comm_size; ++dest) {
        if (dest == my_rank)
            continue;

        MPIw::Send<T>(comm, {T(dest * 3)}, dest, dest * 2);
        const auto [data, status] = MPIw::Recv<T>(comm);
        assert(status.MPI_SOURCE == dest);
        assert(status.MPI_TAG == dest * 5);
        assert(MPIw::Get_count<T>(status) == 1);
        assert(data.size() == 1);
        assert(data[0] == T(dest * 4));
    }
}

template <typename T>
void run_send_recv_multi(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    for (int dest = 0; dest < comm_size; ++dest) {
        if (dest == my_rank)
            continue;

        MPIw::Send<T>(comm, {T(dest * 3)}, dest, dest * 2);
    }

    std::set<int> seen;
    seen.insert(my_rank);

    while (seen.size() != comm_size) {
        const auto [data, status] = MPIw::Recv<T>(comm);
        int src = status.MPI_SOURCE;

        assert(!seen.contains(src));
        assert(status.MPI_TAG == src * 5);
        assert(MPIw::Get_count<T>(status) == 1);
        assert(data.size() == 1);
        assert(data[0] == T(src * 4));
        seen.insert(src);
    }
}

template <typename T>
void run_send_recv_range(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    for (int dest = 0; dest < comm_size; ++dest) {
        if (dest == my_rank)
            continue;

        std::vector<T> to_send(dest);
        std::iota(to_send.begin(), to_send.end(), 0);
        MPIw::Send<T>(comm, to_send, dest, dest * 2);
    }

    std::set<int> seen;
    seen.insert(my_rank);

    while (seen.size() != comm_size) {
        const auto [data, status] = MPIw::Recv<T>(comm);
        int src = status.MPI_SOURCE;
        std::vector<T> expected(src);
        std::iota(expected.begin(), expected.end(), 0);
        std::ranges::transform(expected, expected.begin(),
                               [](auto x) { return T(2) * x; });

        assert(!seen.contains(src));
        assert(status.MPI_TAG == src * 5);
        assert(MPIw::Get_count<T>(status) == src);
        assert(data.size() == src);
        assert(data == expected);
        seen.insert(src);
    }
}

} // namespace master

namespace slave {
template <typename T>
void run_send_recv_one(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);
    const auto [data, status] = MPIw::Recv_one<T>(comm);
    assert(status.MPI_SOURCE == root);
    assert(status.MPI_TAG == my_rank * 2);
    assert(MPIw::Get_count<T>(status) == 1);
    assert(data == T(my_rank * 3));
    MPIw::Send_one<T>(comm, T(my_rank * 4), root, my_rank * 5);
}

template <typename T>
void run_send_recv_single(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);
    const auto [data, status] = MPIw::Recv<T>(comm);
    assert(status.MPI_SOURCE == root);
    assert(status.MPI_TAG == my_rank * 2);
    assert(MPIw::Get_count<T>(status) == 1);
    assert(data.size() == 1);
    assert(data[0] == T(my_rank * 3));
    MPIw::Send<T>(comm, {T(my_rank * 4)}, root, my_rank * 5);
}

template <typename T>
void run_send_recv_multi(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);
    const auto [data, status] = MPIw::Recv<T>(comm);
    assert(status.MPI_SOURCE == root);
    assert(status.MPI_TAG == my_rank * 2);
    assert(MPIw::Get_count<T>(status) == 1);
    assert(data.size() == 1);
    assert(data[0] == T(my_rank * 3));
    MPIw::Send<T>(comm, {T(my_rank * 4)}, root, my_rank * 5);
}

template <typename T>
void run_send_recv_range(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);
    std::vector<T> expected(my_rank);
    std::iota(expected.begin(), expected.end(), 0);

    auto [data, status] = MPIw::Recv<T>(comm);
    assert(status.MPI_SOURCE == root);
    assert(status.MPI_TAG == my_rank * 2);
    assert(MPIw::Get_count<T>(status) == my_rank);
    assert(data.size() == my_rank);
    assert(data == expected);
    std::ranges::transform(data, data.begin(), [](auto x) { return x * T(2); });
    MPIw::Send<T>(comm, data, root, my_rank * 5);
}
} // namespace slave

template <typename T>
void run_send_recv(MPI_Comm comm) {
    int rank = MPIw::Comm_rank(comm);
    int size = MPIw::Comm_size(comm);
    bool is_printing = (rank == 0);

    // one
    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("send / recv one (root: {})", root));
        if (root != rank)
            slave::run_send_recv_one<T>(comm, root);
        else {
            master::run_send_recv_one<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("send / recv one");
    }

    // single
    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("send / recv single (root: {})", root));
        if (root != rank)
            slave::run_send_recv_single<T>(comm, root);
        else {
            master::run_send_recv_single<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("send / recv single");
    }

    // multi
    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("send / recv multi (root: {})", root));
        if (root != rank)
            slave::run_send_recv_multi<T>(comm, root);
        else {
            master::run_send_recv_multi<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("send / recv multi");
    }

    // range
    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(
                fmt::format("send / recv range (root: {})", root));
        if (root != rank)
            slave::run_send_recv_range<T>(comm, root);
        else {
            master::run_send_recv_range<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("send / recv range");
    }
}
