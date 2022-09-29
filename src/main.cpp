#include "include.hpp"
#include <fmt/core.h>
#include <iostream>
#include <mpi.h>

#define print(...) std::cout << fmt::format(__VA_ARGS__) << std::endl;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    char x;
    MPIw::types::get_mpi_type<>(x);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int number;
    int next_rank = (world_rank + 1) % world_size;

    if (world_rank == 0) {
        number = 3;
        MPI_Status stat;
        print("RANK {}: sending '{}' to rank: {}", world_rank, number,
              next_rank);
        MPI_Send(&number, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&number, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print("RANK {}: Recieved number: {}", world_rank, number);
    } else {

        MPI_Recv(&number, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print("RANK {}: Recieved number: {}", world_rank, number);
        number = number * number;
        print("RANK {}: sending '{}' to rank: {}", world_rank, number,
              next_rank);
        MPI_Send(&number, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
