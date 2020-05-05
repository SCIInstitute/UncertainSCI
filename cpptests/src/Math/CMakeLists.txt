project(cppUncertainSCIMath C CXX)

set(SOURCE_FILES
    opoly1d.h
    families.h
    families.cpp
    types.h
    types.cpp
    opoly1d.cpp
)

add_library(cppUncertainSCIMath SHARED STATIC ${SOURCE_FILES})

install(TARGETS cppUncertainSCIMath DESTINATION ${cppUncertainSCI_INSTALL_LIB_DIR})
#install(FILES opoly1d.h families.h types.h DESTINATION ${cppUncertainSCI_INSTALL_INCLUDE_DIR})
