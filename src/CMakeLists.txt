add_executable(test main.cpp)

target_link_libraries(test ${CMAKE_BINARY_DIR}/lib/libgpufaiss.a ${CUDA_LINKER_LIBS})
