#pragma once

#include "../src/include.hpp"
#include "utils.hpp"

namespace master {
template <typename T>
void run_gather_scather_single(MPI_Comm comm) {}
} // namespace master

namespace slave {
template <typename T>
void run_gather_scather_single(MPI_Comm comm, int root) {}

} // namespace slave

template <typename T>
void run_gather_scather(MPI_Comm comm) {}