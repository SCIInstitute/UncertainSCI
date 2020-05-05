project(cppUncertainSCIMath C CXX)

set(SOURCE_FILES
    opoly1d.h
    families.h
    families.cpp
    types.h
    types.cpp
    opoly1d.cpp
)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(cppUncertainSCIMath SHARED STATIC ${SOURCE_FILES})
target_link_libraries(cppUncertainSCIMath Eigen3::Eigen)

install(TARGETS cppUncertainSCIMath DESTINATION ${cppUncertainSCI_INSTALL_LIB_DIR})
#install(FILES opoly1d.h families.h types.h DESTINATION ${cppUncertainSCI_INSTALL_INCLUDE_DIR})
