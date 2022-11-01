project(neuralib)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCES src/*)
file(GLOB HEADERS header/*)

add_executable(neuralib main.cpp)
target_link_libraries(neuralib ${SOURCES} ${HEADERS})