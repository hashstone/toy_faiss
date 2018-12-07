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
#include "../utils/DeviceVector.cuh"
#include "../utils/DeviceTensor.cuh"
#include "../utils/MemorySpace.h"
#include <memory>
#include <thrust/device_vector.h>
#include <vector>

namespace faiss { namespace gpu {

class GpuResources;

/// Base inverted list functionality for IVFFlat and IVFPQ
class IVFBase {
 public:
  IVFBase(GpuResources* resources,
          IndicesOptions indicesOptions,
          MemorySpace space,
          int nlist);

  virtual ~IVFBase();

  /// Reserve GPU memory in our inverted lists for this number of vectors
  void reserveMemory(size_t numVecs);

  /// Clear out all inverted lists, but retain the coarse quantizer
  /// and the product quantizer info
  void reset();

  /// After adding vectors, one can call this to reclaim device memory
  /// to exactly the amount needed. Returns space reclaimed in bytes
  size_t reclaimMemory();

  /// Returns the number of inverted lists
  size_t getNumLists() const;

  /// For debugging purposes, return the list length of a particular
  /// list
  int getListLength(int listId) const;

  /// Return the list indices of a particular list back to the CPU
  std::vector<long> getListIndices(int listId) const;

  int getMaxListLength() const { return maxListLength_; }

 protected:
  /// Reclaim memory consumed on the device for our inverted lists
  /// `exact` means we trim exactly to the memory needed
  size_t reclaimMemory_(bool exact);

  /// Update all device-side list pointer and size information
  void updateDeviceListInfo_(cudaStream_t stream);

  /// For a set of list IDs, update device-side list pointer and size
  /// information
  void updateDeviceListInfo_(const std::vector<int>& listIds,
                             cudaStream_t stream);

  /// Shared function to copy indices from CPU to GPU
  void addIndicesFromCpu_(int listId,
                          const long* indices,
                          size_t numVecs);

 protected:
  /// Collection of GPU resources that we use
  GpuResources* resources_;

  /// Number of inverted lists we maintain
  const int numLists_;

  /// How are user indices stored on the GPU?
  const IndicesOptions indicesOptions_;

  /// What memory space our inverted list storage is in
  const MemorySpace space_;

  /// Device representation of all inverted list index pointers
  /// id -> data
  thrust::device_vector<void*> deviceListIndexPointers_;

  /// Device representation of all inverted list lengths
  /// id -> length
  thrust::device_vector<int> deviceListLengths_;

  /// Maximum list length seen
  int maxListLength_;

  /// Device memory for each separate list, as managed by the host.
  /// Device memory as stored in DeviceVector is stored as unique_ptr
  /// since deviceListSummary_ pointers must remain valid despite
  /// resizing of deviceLists_
  std::vector<std::unique_ptr<DeviceVector<unsigned char>>> deviceListIndices_;
};

} } // namespace
