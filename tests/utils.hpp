#pragma once

#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <fmt/core.h>
#include <iostream>
#include <string>

#define print(format_str, ...)                                                 \
    std::cout << fmt::format(format_str __VA_OPT__(, ) __VA_ARGS__)            \
              << std::endl;

template <typename T>
std::string get_type_name() {
    throw std::runtime_error("Unkonwn type for string conversion");
}
#define register_type_name(type)                                               \
    template <>                                                                \
    std::string get_type_name<type>() {                                        \
        return #type;                                                          \
    }

register_type_name(bool);
register_type_name(char);
register_type_name(signed char);
register_type_name(unsigned char);
register_type_name(int);
register_type_name(unsigned);
register_type_name(long);
register_type_name(unsigned long);
register_type_name(float);
register_type_name(double);
register_type_name(long double);
register_type_name(std::complex<float>);
register_type_name(std::complex<double>);
register_type_name(std::complex<long double>);

void print_start_test(const std::string& name) {
    std::string up_name(name.size(), '\0');
    std::transform(name.begin(), name.end(), up_name.begin(),
                   [](char ch) { return std::toupper(ch); });
    print("{}", std::string(60, '='));
    print("STARTED: {:^45}", up_name);
    print("{}", std::string(60, '='));
}

void print_end_test(const std::string& name) {
    std::string up_name(name.size(), '\0');
    std::transform(name.begin(), name.end(), up_name.begin(),
                   [](char ch) { return std::toupper(ch); });
    print("{}", std::string(60, '='));
    print("ENDED: {:^49}", up_name);
    print("{}", std::string(60, '='));
}

void print_start_section(const std::string& name) {
    std::cout << "Testing: " << name << " ... " << std::flush;
}

void print_end_section(const std::string&) {
    std::cout << "[ OK ]" << std::endl;
}

void print_local_info(MPI_Comm comm) {
    print("Total Comm size: {}\nRank: {}\nProcessor name: {}",
          MPIw::Comm_size(comm), MPIw::Comm_rank(comm),
          MPIw::Get_processor_name());
}