cmake_minimum_required(VERSION 3.2)

project(cppUncertainSCI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(cppUncertainSCI_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(cppUncertainSCI_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(cppUncertainSCI_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(cppUncertainSCI_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/Math)

include_directories(${cppUncertainSCI_INSTALL_INCLUDE_DIR})
include_directories(${cppUncertainSCI_HEADERS_DIR})

add_subdirectory(src)

OPTION(BUILD_TESTING "Build with tests." OFF)
if(BUILD_TESTING)
  add_subdirectory(test)
ENDIF()
