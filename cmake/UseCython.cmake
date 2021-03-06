# This allows to link Cython files
# Examples:
# 1) to compile assembly.pyx to assembly.so:
#   CYTHON_ADD_MODULE(assembly)
# 2) to compile assembly.pyx and something.cpp to assembly.so:
#   CYTHON_ADD_MODULE(assembly something.cpp)

if(NOT CYTHON_INCLUDE_DIRECTORIES)
    set(CYTHON_INCLUDE_DIRECTORIES .)
endif(NOT CYTHON_INCLUDE_DIRECTORIES)

macro(CYTHON_ADD_MODULE name)
    add_custom_command(
        OUTPUT ${name}.cpp
        COMMAND cython
        ARGS -I ${CYTHON_INCLUDE_DIRECTORIES} -o ${name}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${name}.pyx
        DEPENDS ${name}.pyx
        COMMENT "Cythonizing ${name}.pyx")
    add_library(${name} SHARED ${name}.cpp ${ARGN})
    set_target_properties(${name} PROPERTIES PREFIX "")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
endmacro(CYTHON_ADD_MODULE)

