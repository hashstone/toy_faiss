# specify header and cpp files
file(GLOB_RECURSE faiss_gpu_headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE faiss_gpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE faiss_gpu_cuh ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
file(GLOB_RECURSE faiss_gpu_cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

set(CUDA_ARCHS -gencode arch=compute_35,code="compute_35" -gencode arch=compute_52,code="compute_52" -gencode arch=compute_60,code="compute_60")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCHS} -w -Xcompiler -fPIC" )

set(faiss_lib_gpu gpufaiss)

cuda_add_library(${faiss_lib_gpu} STATIC ${faiss_gpu_headers} ${faiss_gpu_cpp} ${faiss_gpu_cuh} ${faiss_gpu_cu})