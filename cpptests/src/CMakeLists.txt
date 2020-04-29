cmake_minimum_required(VERSION 3.2)

project(cppUncertainSCIMathDriver)

add_subdirectory(Math)
set(SOURCE_FILES main.cpp)

add_executable(cppUncertainSCIMathDriver ${SOURCE_FILES})
target_link_libraries(cppUncertainSCIMathDriver cppUncertainSCIMath)
install(TARGETS cppUncertainSCIMathDriver DESTINATION ${cppUncertainSCI_INSTALL_BIN_DIR})
