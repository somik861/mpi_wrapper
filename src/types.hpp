#pragma once

#include <complex>
#include <mpi.h>
#include <stdexcept>

#define MPIw_register_type(cpp_type, mpi_type)                                 \
    namespace MPIw::types {                                                    \
    template <>                                                                \
    inline MPI_Datatype get_mpi_type<cpp_type>(cpp_type) {                     \
        return mpi_type;                                                       \
    }                                                                          \
    }

namespace MPIw::types {

template <typename T>
MPI_Datatype get_mpi_type(T = T{}) {
    throw std::runtime_error(
        "Type is not known by wrapper, please register it in MPIw::types");
}
} // namespace MPIw::types

MPIw_register_type(char, MPI_CHAR);
MPIw_register_type(wchar_t, MPI_WCHAR);
MPIw_register_type(short, MPI_SHORT);
MPIw_register_type(int, MPI_INT);
MPIw_register_type(long, MPI_LONG);
MPIw_register_type(signed char, MPI_SIGNED_CHAR);
MPIw_register_type(unsigned char, MPI_UNSIGNED_CHAR);
MPIw_register_type(unsigned short, MPI_UNSIGNED_SHORT);
MPIw_register_type(unsigned, MPI_UNSIGNED);
MPIw_register_type(unsigned long, MPI_UNSIGNED_LONG);
MPIw_register_type(float, MPI_FLOAT);
MPIw_register_type(double, MPI_DOUBLE);
MPIw_register_type(long double, MPI_LONG_DOUBLE);

MPIw_register_type(bool, MPI_CXX_BOOL);
MPIw_register_type(std::complex<float>, MPI_CXX_COMPLEX);
MPIw_register_type(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX);
MPIw_register_type(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX);
