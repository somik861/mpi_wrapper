#pragma once

#include <algorithm>
#include <cctype>
#include <fmt/core.h>
#include <iostream>
#include <string>

#define print(format_str, ...)                                                 \
    std::cout << fmt::format(format_str __VA_OPT__(, ) __VA_ARGS__)            \
              << std::endl;

void print_start_test(const std::string& name) {
    std::string up_name(name.size(), '\0');
    std::transform(name.begin(), name.end(), up_name.begin(),
                   [](char ch) { return std::toupper(ch); });
    print("{}", std::string(60, '='));
    print("{:^60}", up_name);
    print("{}", std::string(60, '='));
}