/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../GpuIndicesOptions.h"
#include "../utils/Tensor.cuh"
#include <thrust/device_vector.h>

namespace faiss { namespace gpu {

/// Update device-side list pointers in a batch
void runUpdateListPointers(Tensor<int, 1, true>& listIds,
                           Tensor<int, 1, true>& newListLength,
                           Tensor<void*, 1, true>& newIndexPointers,
                           thrust::device_vector<int>& listLengths,
                           thrust::device_vector<void*>& listIndices,
                           cudaStream_t stream);

/// Actually append the new codes / vector indices to the individual lists

/// IVFID append
void runIVFIDInvertedListAppend(Tensor<int, 1, true>& listIds,
                                Tensor<int, 1, true>& listOffset,
                                Tensor<long, 1, true>& indices,
                                thrust::device_vector<void*>& listIndices,
                                IndicesOptions indicesOptions,
                                cudaStream_t stream);

/// IVFID find 
void runIVFIDInvertedListFind(long id,
                              thrust::device_vector<void*>& listIndices,
                              thrust::device_vector<int>& listLengths,
                              IndicesOptions indicesOptions,
                              Tensor<int, 1, true>& offset, // output
                              cudaStream_t stream);

/// IVFID remove
void runIVFIDInvertedListRemove(int listIdx,
                                int listPos,
                                thrust::device_vector<void*>& listIndices,
                                thrust::device_vector<int>& listLengths,
                                IndicesOptions indicesOptions,
                                cudaStream_t stream);

} } // namespace
