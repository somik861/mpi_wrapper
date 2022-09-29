#include "../src/include.hpp"
#include "utils.hpp"
#include <cassert>

void get_local_info() {}

template <typename T>
void run_all_test();

template <typename T>
void run_send_recv();

int main(int argc, char** argv) {
    MPIw::Init_raii(&argc, &argv);
    run_all_test<int>();

    print("All tests passed !!!");
}

template <typename T>
void run_all_test() {
    run_send_recv<T>();
}

template <typename T>
void run_send_recv() {
    print_start_test("send / recv");

    print_end_test("send / recv");
}