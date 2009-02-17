project(schroedinger)

cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(UseCython)

find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)
set(HERMES2D_PXD_INCLUDE /home/ondra/lib/lib/python/hermes2d/include)
set(HERMES2D_INCLUDE /home/ondra/lib/lib/python/hermes2d/include)

include_directories(
    ${PYTHON_INCLUDE_PATH}
    ${NUMPY_INCLUDE_PATH}
    ${HERMES2D_INCLUDE}
    )
set(CYTHON_INCLUDE_DIRECTORIES ${HERMES2D_PXD_INCLUDE})
cython_add_module(cschroed dft)
#target_link_libraries(hermes2d hermes)
