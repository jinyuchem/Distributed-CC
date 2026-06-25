cmake_minimum_required(VERSION 3.15)

project(Distributed-CC LANGUAGES C CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
  message(FATAL_ERROR "Distributed-CC requires g++ version 10 or newer for the symmetrized tensor contraction implementation")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DISTR_CC_NATIVE "Enable native CPU tuning such as -march=native" OFF)
option(DISTR_CC_OPENMP "Enable OpenMP for native helper kernels" OFF)

if(DISTR_CC_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS C)
endif()

add_library(distr_cc SHARED
  src/rccsdt.c
  src/rccsdt_q.c
  src/rccsdtq.c
)

set_target_properties(distr_cc PROPERTIES
  PREFIX ""
  OUTPUT_NAME "distr_cc"
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(MSVC)
  target_compile_options(distr_cc PRIVATE /O2)
else()
  target_compile_options(distr_cc PRIVATE -O3)
  if(DISTR_CC_NATIVE)
    target_compile_options(distr_cc PRIVATE -march=native)
  endif()
endif()

if(DISTR_CC_OPENMP)
  target_link_libraries(distr_cc PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS distr_cc
  LIBRARY DESTINATION distr_cc
  RUNTIME DESTINATION distr_cc
  ARCHIVE DESTINATION distr_cc
)
