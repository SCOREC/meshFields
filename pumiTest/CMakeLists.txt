cmake_minimum_required(VERSION 3.16)
project(PumiTest LANGUAGES CXX C)

find_package(MPI REQUIRED)

find_package(SCOREC REQUIRED)

add_executable(PumiTest main.cpp)

target_link_libraries(PumiTest PRIVATE
    MPI::MPI_CXX
    SCOREC::apf
    SCOREC::mds
    SCOREC::gmi
    SCOREC::pcu
    SCOREC::lion
)
