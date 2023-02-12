project(neuralib)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

find_package(Boost 1.58.0 COMPONENTS filesystem system iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

file(GLOB SOURCES src/*)
file(GLOB HEADERS header/*)

add_executable(neuralib main.cpp)
target_link_libraries(neuralib ${SOURCES} ${HEADERS} ${Boost_LIBRARIES})