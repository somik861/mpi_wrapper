#include "../src/include.hpp"
#include "allgather.hpp"
#include "broadcast.hpp"
#include "gather_scatter.hpp"
#include "reduce.hpp"
#include "send_recv.hpp"
#include <cassert>

template <typename T>
void run_all_test();

int main(int argc, char** argv) {
    MPIw::Init_raii _mpi_init(&argc, &argv);
    run_all_test<int>();
    run_all_test<long>();
    run_all_test<unsigned>();
    run_all_test<unsigned long>();
    run_all_test<float>();
    run_all_test<double>();
    run_all_test<long double>();

    if (MPIw::Comm_rank(MPI_COMM_WORLD) == 0)
        print("All tests passed !!!");
}

template <typename T>
void run_all_test() {
    bool is_printing = (MPIw::Comm_rank(MPI_COMM_WORLD) == 0);
    std::string test_name = fmt::format("ALL TESTS ({})", get_type_name<T>());

    if (is_printing) {
        print_start_test(test_name);
        print_local_info(MPI_COMM_WORLD);
    }

    run_send_recv<T>(MPI_COMM_WORLD);
    run_broadcast<T>(MPI_COMM_WORLD);
    run_gather_scather<T>(MPI_COMM_WORLD);
    run_allgather<T>(MPI_COMM_WORLD);
    run_reduce<T>(MPI_COMM_WORLD);

    if (is_printing)
        print_end_test(test_name);
}
