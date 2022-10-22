#pragma once

#include <concepts>
#include <type_traits>

namespace MPIw::details::cnpts {
template <typename T>
concept EnumOrInt = requires(T) {
	requires std::is_enum_v<T> || std::is_same_v<T, int>;
};
} // namespace MPIw::details::cnpts