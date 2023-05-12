project(neuralib)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -march=znver2 -ffast-math -mavx2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fopenmp")


find_package(Boost 1.58.0 COMPONENTS filesystem system iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

file(GLOB SOURCES src/*)
file(GLOB SOURCES_MAIN main/*.cpp)
list(APPEND SOURCES ${SOURCES_MAIN})

file(GLOB HEADERS header/*)
file(GLOB HEADERS_MAIN main/main.hpp)
list(APPEND HEADERS ${HEADERS_MAIN})
file(GLOB LIBS libs/*)
list(APPEND HEADERS ${LIBS})

add_executable(neuralib main.cpp)
target_link_libraries(neuralib ${SOURCES} ${HEADERS} ${Boost_LIBRARIES})