# S2HAT

# Top-level CMakeLists file
# - contains the call to the project() command and some global settings
# - search for external dependencies: MPI, OpenMP, FFTW3
# - add relevant subdirectories to the build

# ----------------------------------------------------------------------------------------#
# General configuration
# ----------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.22...3.25)
enable_language( Fortran )

# Project name and a few useful settings
project(Xpure
        VERSION 2.56
        DESCRIPTION "Pure pseudo-power spectrum estimation"
        LANGUAGES C Fortran)

# Enforce language standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_Fortran_STANDARD 77)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# We are building libraries that will eventually be linked into shared modules.
# All code should be built with PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Auxiliary files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# ----------------------------------------------------------------------------------------#
# External packages
# ----------------------------------------------------------------------------------------#

# Search for MPI 
find_package(MPI REQUIRED)
find_package(MPI COMPONENTS Fortran REQUIRED)

# Search for CFITSIO
find_package(CFITSIO REQUIRED)

# Search for FFTW
find_package(FFTW REQUIRED)

# Search for HEALPIX
find_package(HEALPIX REQUIRED)

# Search for S2HAT
find_package(S2HAT REQUIRED)

# BLAS/LAPACK
find_package(BLAS)
find_package(LAPACK REQUIRED)

# ----------------------------------------------------------------------------------------#
# Subdirectories
# ----------------------------------------------------------------------------------------#

add_subdirectory(src)
