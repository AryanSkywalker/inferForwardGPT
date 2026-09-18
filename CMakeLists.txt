cmake_minimum_required(VERSION 3.16)
project(forwardGPT2 CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O2 -Wextra -Wall)
add_executable(gpt2 src/main.cpp src/weights.cpp)
target_include_directories(gpt2 PRIVATE include)

