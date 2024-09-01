cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(distributed-macs)

set(CMAKE_CXX_STANDARD 14)

# Set the path to libtorch
set(TORCH_PATH "/opt/homebrew/opt/pytorch")

# Find the torch package
find_package(Torch REQUIRED PATHS "${TORCH_PATH}")

# Find pybind11
find_package(pybind11 REQUIRED)

# Add your source files
file(GLOB SOURCES "src/*.cpp")
add_executable(your_executable ${SOURCES})

# Link against libtorch and pybind11
target_link_libraries(your_executable "${TORCH_LIBRARIES}" pybind11::pybind11)

# Include directories
target_include_directories(your_executable PRIVATE "${TORCH_INCLUDE_DIRS}")