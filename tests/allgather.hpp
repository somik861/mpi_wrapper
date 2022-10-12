#pragma once

#include "../src/include.hpp"
#include "utils.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace master {
template <typename T>
void run_allgather_single(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    std::vector<T> to_send = {T(my_rank * 2)};
    std::vector<T> expected(comm_size);
    std::iota(expected.begin(), expected.end(), 0);
    std::ranges::transform(expected, expected.begin(),
                           [](auto x) { return x * 2; });

    auto got = MPIw::Allgather(comm, to_send);
    assert(got == expected);
}

template <typename T>
void run_allgather_range(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);
    int comm_size = MPIw::Comm_size(comm);

    std::vector<T> to_send(comm_size);
    std::iota(to_send.begin(), to_send.end(), my_rank * comm_size);

    std::vector<T> expected(comm_size * comm_size);
    std::iota(expected.begin(), expected.end(), 0);

    auto got = MPIw::Allgather(comm, to_send);
    assert(got == expected);
}

template <typename T>
void run_allgather_variable(MPI_Comm comm) {
    int my_rank = MPIw::Comm_rank(comm);

    for (int n = 0; n < 10; ++n) {
        int size = std::random_device{}() % 100;
        std::vector<T> to_send(size);
        std::iota(to_send.begin(), to_send.end(), my_rank);

        auto got = MPIw::Allgatherv(comm, to_send);
        assert(got[my_rank] == to_send);
    }
}
} // namespace master

template <typename T>
void run_allgather(MPI_Comm comm) {
    int rank = MPIw::Comm_rank(comm);
    bool is_printing = (rank == 0);

    if (is_printing)
        print_start_section("allgather single");
    master::run_allgather_single<T>(comm);
    MPIw::Barrier(comm);
    if (is_printing)
        print_end_section("allgather single");

    if (is_printing)
        print_start_section("allgather range");
    master::run_allgather_range<T>(comm);
    MPIw::Barrier(comm);
    if (is_printing)
        print_end_section("allgather range");

    if (is_printing)
        print_start_section("allgather variable");
    master::run_allgather_variable<T>(comm);
    MPIw::Barrier(comm);
    if (is_printing)
        print_end_section("allgather variable");
}