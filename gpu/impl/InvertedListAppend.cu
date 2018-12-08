/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "InvertedListAppend.cuh"
#include "../utils/FaissAssert.h"
#include "../utils/Float16.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/Tensor.cuh"
#include "../utils/StaticUtils.h"

#include <iostream>

namespace faiss { namespace gpu {

__global__ void
runUpdateListPointers(Tensor<int, 1, true> listIds,
                      Tensor<int, 1, true> newListLength,
                      Tensor<void*, 1, true> newIndexPointers,
                      int* listLengths,
                      void** listIndices) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= listIds.getSize(0)) {
    return;
  }

  int listId = listIds[index];
  listLengths[listId] = newListLength[index];
  listIndices[listId] = newIndexPointers[index];
}

void
runUpdateListPointers(Tensor<int, 1, true>& listIds,
                      Tensor<int, 1, true>& newListLength,
                      Tensor<void*, 1, true>& newIndexPointers,
                      thrust::device_vector<int>& listLengths,
                      thrust::device_vector<void*>& listIndices,
                      cudaStream_t stream) {
  int numThreads = std::min(listIds.getSize(0), getMaxThreadsCurrentDevice());
  int numBlocks = utils::divUp(listIds.getSize(0), numThreads);

  dim3 grid(numBlocks);
  dim3 block(numThreads);

  runUpdateListPointers<<<grid, block, 0, stream>>>(
    listIds, newListLength, newIndexPointers,
    listLengths.data().get(),
    listIndices.data().get());

  CUDA_TEST_ERROR();
}

template <IndicesOptions Opt>
__global__ void
ivfpqInvertedListAppend(Tensor<int, 1, true> listIds,
                        Tensor<int, 1, true> listOffset,
                        Tensor<long, 1, true> indices,
                        void** listIndices) {
  int encodingToAdd = blockIdx.x * blockDim.x + threadIdx.x;

  if (encodingToAdd >= listIds.getSize(0)) {
    return;
  }

  int listId = listIds[encodingToAdd];
  int offset = listOffset[encodingToAdd];

  // Add vector could be invalid (contains NaNs etc)
  if (listId == -1 || offset == -1) {
    return;
  }

  long index = indices[encodingToAdd];

  if (Opt == INDICES_32_BIT) {
    // FIXME: there could be overflow here, but where should we check this?
    ((int*) listIndices[listId])[offset] = (int) index;
  } else if (Opt == INDICES_64_BIT) {
    ((long*) listIndices[listId])[offset] = (long) index;
  } else {
    // INDICES_CPU or INDICES_IVF; no indices are being stored
  }
}

void
runIVFIDInvertedListAppend(Tensor<int, 1, true>& listIds,
                           Tensor<int, 1, true>& listOffset,
                           Tensor<long, 1, true>& indices,
                           thrust::device_vector<void*>& listIndices,
                           IndicesOptions indicesOptions,
                           cudaStream_t stream) {
  int numThreads = std::min(listIds.getSize(0), getMaxThreadsCurrentDevice());
  int numBlocks = utils::divUp(listIds.getSize(0), numThreads);

  dim3 grid(numBlocks);
  dim3 block(numThreads);

#define RUN_APPEND(IND)                                         \
  do {                                                          \
    ivfpqInvertedListAppend<IND><<<grid, block, 0, stream>>>(   \
      listIds, listOffset, indices,                  \
      listIndices.data().get());                                \
  } while (0)

  if ((indicesOptions == INDICES_CPU) || (indicesOptions == INDICES_IVF)) {
    // no need to maintain indices on the GPU
    RUN_APPEND(INDICES_IVF);
  } else if (indicesOptions == INDICES_32_BIT) {
    RUN_APPEND(INDICES_32_BIT);
  } else if (indicesOptions == INDICES_64_BIT) {
    RUN_APPEND(INDICES_64_BIT);
  } else {
    // unknown index storage type
    FAISS_ASSERT(false);
  }

  CUDA_TEST_ERROR();

#undef RUN_APPEND
}

template <IndicesOptions Opt>
__global__ void
ivfpqInvertedListFind(long id,
                      int nlist,
                      void** listIndices,
                      int* listLengths,
                      Tensor<int, 1, true> offset) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nlist) {
    return;
  }

  // FIXME: loop in kernel? 
  offset[idx] = -1;
  int total = listLengths[idx];
  if (Opt == INDICES_32_BIT) {
    int *indice = (int*) (listIndices[idx]);
    for (int i = 0; i < total; ++i)
    {
      if (indice[i] == (int)id)
      {
        offset[idx] = i;
        break;
      }
    } 
  } else if (Opt == INDICES_64_BIT) {
    long *indice = (long*) (listIndices[idx]);
    for (int i = 0; i < total; ++i)
    {
      if (indice[i] == id)
      {
        offset[idx] = i;
        break;
      }
    } 
  } else {
    // INDICES_CPU or INDICES_IVF; no indices are being stored
  }
}

void runIVFIDInvertedListFind(long id,
                              thrust::device_vector<void*>& listIndices,
                              thrust::device_vector<int>& listLengths,
                              IndicesOptions indicesOptions,
                              Tensor<int, 1, true>& offset, // output
                              cudaStream_t stream)
{
  int nlist = (int)listIndices.size();
  int numThreads = std::min(nlist, getMaxThreadsCurrentDevice());
  int numBlocks = utils::divUp(nlist, numThreads);

  dim3 grid(numBlocks);
  dim3 block(numThreads);

#define RUN_FIND(IND)                                      \
  do {                                                     \
    ivfpqInvertedListFind<IND><<<grid, block, 0, stream>>>(\
      id, \
      offset.getSize(0), \
      listIndices.data().get(), \
      listLengths.data().get(), \
      offset);     \
  } while (0)

  if ((indicesOptions == INDICES_CPU) || (indicesOptions == INDICES_IVF)) {
    // no need to maintain indices on the GPU
    RUN_FIND(INDICES_IVF);
  } else if (indicesOptions == INDICES_32_BIT) {
    RUN_FIND(INDICES_32_BIT);
  } else if (indicesOptions == INDICES_64_BIT) {
    RUN_FIND(INDICES_64_BIT);
  } else {
    // unknown index storage type
    FAISS_ASSERT(false);
  }
  CUDA_TEST_ERROR();

#undef RUN_FIND
}

// only use one thread to delete id
template <IndicesOptions Opt>
__global__ void
ivfpqInvertedListRemove(int listIdx,
                        int listPos,
                        void** listIndices,
                        int* listLengths) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx > 1) {
    return;
  }

  // use last element fit in the position you want deleted
  if (Opt == INDICES_32_BIT) {
    int *indice = (int*) listIndices[listIdx];
    indice[listPos] = indice[listLengths[listIdx] - 1];
  } else if (Opt == INDICES_64_BIT) {
    long *indice = (long *) listIndices[listIdx];
    indice[listPos] = indice[listLengths[listIdx] - 1];
  } else {
    // INDICES_CPU or INDICES_IVF; no indices are being stored
  }
}

void runIVFIDInvertedListRemove(int listIdx,
                                int listPos,
                                thrust::device_vector<void*>& listIndices,
                                thrust::device_vector<int>& listLengths,
                                IndicesOptions indicesOptions,
                                cudaStream_t stream)
{
#define RUN_REMOVE(IND)                               \
  do {                                                \
    ivfpqInvertedListRemove<IND><<<1, 1, 0, stream>>>(\
      listIdx, \
      listPos, \
      listIndices.data().get(), \
      listLengths.data().get());\
  } while (0)

  if ((indicesOptions == INDICES_CPU) || (indicesOptions == INDICES_IVF)) {
    // no need to maintain indices on the GPU
    RUN_REMOVE(INDICES_IVF);
  } else if (indicesOptions == INDICES_32_BIT) {
    RUN_REMOVE(INDICES_32_BIT);
  } else if (indicesOptions == INDICES_64_BIT) {
    RUN_REMOVE(INDICES_64_BIT);
  } else {
    // unknown index storage type
    FAISS_ASSERT(false);
  }
  CUDA_TEST_ERROR();

#undef RUN_FIND
}

} } // namespace
