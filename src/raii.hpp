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

class Init_threads_raii {
  public:
	Init_threads_raii(int* argc, char*** argv, int required) {
		errors::check_code(MPI_Init_thread(argc, argv, required, &_provided));
	}

	Init_threads_raii(const Init_threads_raii&) = delete;
	Init_threads_raii& operator=(const Init_threads_raii&) = delete;

	Init_threads_raii(Init_threads_raii&&) = delete;
	Init_threads_raii&& operator=(Init_threads_raii&&) = delete;

	~Init_threads_raii() { errors::check_code(MPI_Finalize()); }

	int support_level() const { return _provided; }

  private:
	int _provided;
};

class Comm_raii {
  public:
	MPI_Comm comm = MPI_COMM_NULL;

	Comm_raii() = default;
	Comm_raii(const Comm_raii&) = delete;
	Comm_raii& operator=(const Comm_raii&) = delete;

	Comm_raii(Comm_raii&&) = delete;
	Comm_raii&& operator=(Comm_raii&&) = delete;

	~Comm_raii() {
		if (comm != MPI_COMM_NULL)
			errors::check_code(MPI_Comm_free(&comm));
	}

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

	~Group_raii() {
		if (group != MPI_GROUP_NULL)
			errors::check_code(MPI_Group_free(&group));
	}

	MPI_Group& get() { return group; }
	operator MPI_Group() { return group; }
	MPI_Group* operator&() { return &group; }
};

class Type_raii {
  public:
	MPI_Datatype type = MPI_DATATYPE_NULL;

	Type_raii() = default;
	Type_raii(const Type_raii&) = delete;
	Type_raii& operator=(const Type_raii&) = delete;

	Type_raii(Type_raii&&) = delete;
	Type_raii&& operator=(Type_raii&&) = delete;

	~Type_raii() {
		if (type != MPI_DATATYPE_NULL)
			errors::check_code(MPI_Type_free(&type));
	}

	MPI_Datatype& get() { return type; }
	operator MPI_Datatype() { return type; }
	MPI_Datatype* operator&() { return &type; }
};
} // namespace MPIw
