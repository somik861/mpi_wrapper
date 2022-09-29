#pragma once

#include "../src/include.hpp"
#include "utils.hpp"

namespace master {
template <typename T>
void run_send_recv_single(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    for (int dest = 0; dest < comm_size; ++dest) {
        if (dest == my_rank)
            continue;

        MPIw::Send<T>(comm, {dest * 3}, dest, dest * 2);
        const auto [data, status] = MPIw::Recv<T>(comm);
        assert(status.MPI_SOURCE == dest);
        assert(status.MPI_TAG == dest * 5);
        assert(MPIw::Get_count<T>(status) == 1);
        assert(data.size() == 1);
        assert(data[0] == dest * 4);
    }
}
} // namespace master

namespace slave {
template <typename T>
void run_send_recv_single(MPI_Comm comm, int root) {
    int my_rank = MPIw::Comm_rank(comm);
    const auto [data, status] = MPIw::Recv<T>(comm);
    assert(status.MPI_SOURCE == root);
    assert(status.MPI_TAG == my_rank * 2);
    assert(MPIw::Get_count<T>(status) == 1);
    assert(data.size() == 1);
    assert(data[0] == my_rank * 3);
    MPIw::Send<T>(comm, {my_rank * 4}, root, my_rank * 5);
}
} // namespace slave

template <typename T>
void run_send_recv(MPI_Comm comm) {
    int rank = MPIw::Comm_rank(comm);
    int size = MPIw::Comm_size(comm);
    bool is_printing = (rank == 0);

    for (int root = 0; root < size; ++root) {
        if (is_printing)
            print_start_section(fmt::format("send / recv (root: {})", root));
        if (root != rank)
            slave::run_send_recv_single<T>(comm, root);
        else {
            master::run_send_recv_single<T>(comm);
        }
        MPIw::Barrier(comm);
        if (is_printing)
            print_end_section("send / recv");
    }
}
