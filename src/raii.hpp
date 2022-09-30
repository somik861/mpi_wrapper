#pragma once

#include <mpi.h>

namespace MPIw {
class Init_raii {
  public:
    Init_raii(int* argc, char*** argv) {
        errors::check_code(MPI_Init(argc, argv));
    }

    Init_raii(const Init_raii&) = delete;
    Init_raii& operator=(const Init_raii&) = delete;

    Init_raii(Init_raii&&) = delete;
    Init_raii&& operator=(Init_raii&&) = delete;

    ~Init_raii() { errors::check_code(MPI_Finalize()); }
};

class Comm_raii {
  public:
    MPI_Comm comm = MPI_COMM_NULL;

    Comm_raii() = default;
    Comm_raii(const Comm_raii&) = delete;
    Comm_raii& operator=(const Comm_raii&) = delete;

    Comm_raii(Comm_raii&&) = delete;
    Comm_raii&& operator=(Comm_raii&&) = delete;

    ~Comm_raii() { errors::check_code(MPI_Comm_free(&comm)); }

    MPI_Comm& get() { return comm; }
    operator MPI_Comm() { return comm; }
    MPI_Comm* operator&() { return &comm; }
};

class Group_raii {
  public:
    MPI_Group group = MPI_GROUP_NULL;

    Group_raii() = default;
    Group_raii(const Group_raii&) = delete;
    Group_raii& operator=(const Group_raii&) = delete;

    Group_raii(Group_raii&&) = delete;
    Group_raii&& operator=(Group_raii&&) = delete;

    ~Group_raii() { errors::check_code(MPI_Group_free(&group)); }

    MPI_Group& get() { return group; }
    operator MPI_Group() { return group; }
    MPI_Group* operator&() { return &group; }
};
} // namespace MPIw