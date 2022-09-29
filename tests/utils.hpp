#pragma once

#include <fmt/core.h>
#include <string>

#define print(format, ...)                                                     \
    std::cout << fmt::format(format __VA_OPT__(,) __VA_ARGS__) << std::endl;

void print_start_test(const std::string& name)
{
    print(std::string("=", 20));
    print("{:^20}", name);
    print(std::string("=", 20));
}