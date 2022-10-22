#pragma once

#include <concepts>
#include <iterator>
#include <type_traits>

namespace MPIw::details::cnpts {
template <typename T>
concept EnumOrInt = requires(T) {
	requires std::is_enum_v<T> || std::is_same_v<T, int>;
};

template <typename T>
concept Container = requires(T a) {

	{ a.begin() } -> std::contiguous_iterator;
	{ a.size() } -> std::same_as<std::size_t>;
	{ sizeof(typename T::value_type) } -> std::same_as<std::size_t>;
};
} // namespace MPIw::details::cnpts