cmake_minimum_required(VERSION 2.6)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -O2")
add_library(listrankmf listrankmf.cpp)
INSTALL(TARGETS listrankmf
        ARCHIVE DESTINATION lib)
INSTALL(FILES listrankmf.h
        DESTINATION include)
