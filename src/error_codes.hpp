#pragma once

#include "structs.hpp"
#include <mpi.h>
#include <source_location>
#include <string>

namespace MPIw::errors {
inline std::string error_message(int error_code) {
    switch (error_code) {
    case MPI_SUCCESS:
        return "No error";
    case MPI_ERR_BUFFER:
        return "Invalid buffer pointer";
    case MPI_ERR_COUNT:
        return "Invalid count argument";
    case MPI_ERR_TYPE:
        return "Invalid datatype argument";
    case MPI_ERR_TAG:
        return "Invalid tag argument";
    case MPI_ERR_COMM:
        return "Invalid communicator";
    case MPI_ERR_RANK:
        return "Invalid rank";
    case MPI_ERR_REQUEST:
        return "Invalid request (handle)";
    case MPI_ERR_ROOT:
        return "Invalid root";
    case MPI_ERR_GROUP:
        return "Invalid group";
    case MPI_ERR_OP:
        return "Invalid operation";
    case MPI_ERR_TOPOLOGY:
        return "Invalid topology";
    case MPI_ERR_DIMS:
        return "Invalid dimension argument";
    case MPI_ERR_ARG:
        return "Invalid argument of some other kind";
    case MPI_ERR_UNKNOWN:
        return "Unknown error";
    case MPI_ERR_TRUNCATE:
        return "Message truncated on receive";
    case MPI_ERR_OTHER:
        return "Known error not in this list";
    case MPI_ERR_INTERN:
        return "Internal MPI (implementation) error";
    case MPI_ERR_IN_STATUS:
        return "Error code is in status";
    case MPI_ERR_PENDING:
        return "Pending request";
    case MPI_ERR_KEYVAL:
        return "Invalid keyval has been passed";
    case MPI_ERR_NO_MEM:
        return "MPI_ALLOC_MEM failed because memory is exhausted";
    case MPI_ERR_BASE:
        return "Invalid base passed to MPI_FREE_MEM";
    case MPI_ERR_INFO_KEY:
        return "Key longer than MPI_MAX_INFO_KEY";
    case MPI_ERR_INFO_VALUE:
        return "Value longer than MPI_MAX_INFO_VAL";
    case MPI_ERR_INFO_NOKEY:
        return "Invalid key passed to MPI_INFO_DELETE";
    case MPI_ERR_SPAWN:
        return "Error in spawning processes";
    case MPI_ERR_PORT:
        return "Invalid port name passed to MPI_COMM_CONNECT";
    case MPI_ERR_SERVICE:
        return "Invalid service name passed to MPI_UNPUBLISH_NAME";
    case MPI_ERR_NAME:
        return "Invalid service name passed to MPI_LOOKUP_NAME";
    case MPI_ERR_WIN:
        return "Invalid win argument";
    case MPI_ERR_SIZE:
        return "Invalid size argument";
    case MPI_ERR_DISP:
        return "Invalid disp argument";
    case MPI_ERR_INFO:
        return "Invalid info argument";
    case MPI_ERR_LOCKTYPE:
        return "Invalid locktype argument";
    case MPI_ERR_ASSERT:
        return "Invalid assert argument";
    case MPI_ERR_RMA_CONFLICT:
        return "Conflicting accesses to window";
    case MPI_ERR_RMA_SYNC:
        return "Wrong synchronization of RMA calls";
    default:
        return std::string("Unknown error code: ") + std::to_string(error_code);
    }
}

inline void check_code(
    int error_code,
    const std::source_location& location = std::source_location::current()) {
    if (error_code == MPI_SUCCESS)
        return;

    throw std::runtime_error(std::string("MPI_ERROR: ") +
                             error_message(error_code) +
                             "\nCalled by: " + location.file_name() +
                             "::" + location.function_name() + " at " +
                             std::to_string(location.line()) + "\n");
}

} // namespace MPIw::errors