# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(GA_solver LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Project name and version
project(GA_SOLVER LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable CUDA
enable_language(CUDA)

# Set build type to Release by default (fixes ptxas debug issue)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Separate debug flags for host and device code
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Host debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    # Device debug flags (no -G flag to avoid ptxas error)
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -O0")
else()
    # Release optimization flags
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Define data directories
set(DATA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/data")
set(DATA_BINARY_DIR "${CMAKE_BINARY_DIR}/data")

# Create binary data directory
file(MAKE_DIRECTORY ${DATA_BINARY_DIR})

# Find all CSV files in the data directory
file(GLOB DATA_FILES "${DATA_SOURCE_DIR}/*.csv")

# Copy each data file to the binary directory
foreach(DATA_FILE ${DATA_FILES})
    get_filename_component(FILENAME ${DATA_FILE} NAME)
    configure_file(${DATA_FILE} ${DATA_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

# Create logs directory if it doesn't exist
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/logs")

# Define DATA_DIR for the source code
add_compile_definitions(DATA_DIR="${DATA_BINARY_DIR}")
add_compile_definitions(LOGS_DIR="${CMAKE_SOURCE_DIR}/logs")

# Add source files - Updated for EAX implementation
set(SRC_FILES
    src/genome.cu
    src/population.cu
    src/cost_matrix.cu
    src/fitness_evaluator.cu
    src/parent_selection.cu
    src/crossover.cu                # Now contains EAX implementation
    src/eax_cost_integration.cu     # New: Cost-aware EAX utilities
    src/mutation.cu
)

# Include header files
include_directories(include)

# Create a library for shared logic
add_library(ga_solver_lib ${SRC_FILES})
target_include_directories(ga_solver_lib PRIVATE include)

# Link CUDA libraries and required dependencies
target_link_libraries(ga_solver_lib PRIVATE 
    CUDA::cudart
    CUDA::curand
)

# Set CUDA architectures for better performance (auto-detect)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()
set_property(TARGET ga_solver_lib PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# Add compiler-specific optimizations (safe flags only)
target_compile_options(ga_solver_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --maxrregcount=64
    >
)

# Add release-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(ga_solver_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3>
    )
endif()

# Add executable for the main application
add_executable(ga_solver src/ga_solver_main.cu)
target_link_libraries(ga_solver PUBLIC ga_solver_lib)

# Set CUDA architectures for the executable
set_property(TARGET ga_solver PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# Add memory optimization flags
target_compile_definitions(ga_solver_lib PRIVATE
    MAX_CITIES=256
    MAX_CYCLES=64
    MAX_CYCLE_LENGTH=64
)

# Optional: Add performance profiling support
option(ENABLE_PROFILING "Enable CUDA profiling" OFF)
if(ENABLE_PROFILING)
    target_compile_definitions(ga_solver_lib PRIVATE ENABLE_PROFILING=1)
    target_compile_options(ga_solver_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    )
endif()

# Optional: Add memory debugging (only in debug mode)
option(ENABLE_CUDA_MEMCHECK "Enable CUDA memory checking" OFF)
if(ENABLE_CUDA_MEMCHECK AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(ga_solver_lib PRIVATE CUDA_MEMCHECK=1)
endif()

# Print configuration information
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")

# Enable testing (commented out as in original)
# enable_testing()

# Add test for EAX Crossover
# add_executable(test_eax_crossover tests/test_eax_crossover.cu)
# target_link_libraries(test_eax_crossover PRIVATE ga_solver_lib)
# add_test(NAME EAXCrossoverTest COMMAND test_eax_crossover)