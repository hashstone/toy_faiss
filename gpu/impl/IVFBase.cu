/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "IVFBase.cuh"
#include "../GpuResources.h"
#include "InvertedListAppend.cuh"
#include "../utils/DeviceDefs.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/HostTensor.cuh"
#include <limits>
#include <thrust/host_vector.h>
#include <unordered_map>

namespace faiss { namespace gpu {

IVFBase::IVFBase(GpuResources* resources,
                 IndicesOptions indicesOptions,
                 MemorySpace space,
                 int nlist) :
    resources_(resources),
    indicesOptions_(indicesOptions),
    space_(space),
    numLists_(nlist),
    maxListLength_(0) {
  reset();
}

IVFBase::~IVFBase() {
}

void
IVFBase::reserveMemory(size_t numVecs) {
  size_t vecsPerList = numVecs / deviceListIndices_.size();
  if (vecsPerList < 1) {
    return;
  }

  auto stream = resources_->getDefaultStreamCurrentDevice();

  if ((indicesOptions_ == INDICES_32_BIT) ||
      (indicesOptions_ == INDICES_64_BIT)) {
    // Reserve for index lists as well
    size_t bytesPerIndexList = vecsPerList *
      (indicesOptions_ == INDICES_32_BIT ? sizeof(int) : sizeof(long));

    for (auto& list : deviceListIndices_) {
      list->reserve(bytesPerIndexList, stream);
    }
  }

  // Update device info for all lists, since the base pointers may
  // have changed
  updateDeviceListInfo_(stream);
}

void
IVFBase::reset() {
  deviceListIndices_.clear();
  deviceListIndexPointers_.clear();
  deviceListLengths_.clear();

  for (size_t i = 0; i < numLists_; ++i) {
    deviceListIndices_.emplace_back(
      std::unique_ptr<DeviceVector<unsigned char>>(
        new DeviceVector<unsigned char>(space_)));
  }

  deviceListIndexPointers_.resize(numLists_, nullptr);
  deviceListLengths_.resize(numLists_, 0);
  maxListLength_ = 0;
}

size_t
IVFBase::reclaimMemory() {
  // Reclaim all unused memory exactly
  return reclaimMemory_(true);
}

size_t
IVFBase::reclaimMemory_(bool exact) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  size_t totalReclaimed = 0;

  for (int i = 0; i < deviceListIndices_.size(); ++i) {
    auto& indices = deviceListIndices_[i];
    totalReclaimed += indices->reclaim(exact, stream);

    deviceListIndexPointers_[i] = indices->data();
  }

  // Update device info for all lists, since the base pointers may
  // have changed
  updateDeviceListInfo_(stream);

  return totalReclaimed;
}

void
IVFBase::updateDeviceListInfo_(cudaStream_t stream) {
  std::vector<int> listIds(deviceListIndices_.size());
  for (int i = 0; i < deviceListIndices_.size(); ++i) {
    listIds[i] = i;
  }

  updateDeviceListInfo_(listIds, stream);
}

void
IVFBase::updateDeviceListInfo_(const std::vector<int>& listIds,
                               cudaStream_t stream) {
  auto& mem = resources_->getMemoryManagerCurrentDevice();

  HostTensor<int, 1, true>
    hostListsToUpdate({(int) listIds.size()});
  HostTensor<int, 1, true>
    hostNewListLength({(int) listIds.size()});
  HostTensor<void*, 1, true>
    hostNewIndexPointers({(int) listIds.size()});

  size_t indiceSize =
    (indicesOptions_ == INDICES_32_BIT ? sizeof(int) : sizeof(long));

  for (int i = 0; i < listIds.size(); ++i) {
    auto listId = listIds[i];
    auto& indices = deviceListIndices_[listId];

    hostListsToUpdate[i] = listId;
    hostNewListLength[i] = indices->size() / indiceSize;
    hostNewIndexPointers[i] = indices->data();
  }

  // Copy the above update sets to the GPU
  DeviceTensor<int, 1, true> listsToUpdate(
    mem, hostListsToUpdate, stream);
  DeviceTensor<int, 1, true> newListLength(
    mem,  hostNewListLength, stream);
  DeviceTensor<void*, 1, true> newIndexPointers(
    mem, hostNewIndexPointers, stream);

  // Update all pointers to the lists on the device that may have
  // changed
  runUpdateListPointers(listsToUpdate,
                        newListLength,
                        newIndexPointers,
                        deviceListLengths_,
                        deviceListIndexPointers_,
                        stream);
}

size_t
IVFBase::getNumLists() const {
  return numLists_;
}

int
IVFBase::getListLength(int listId) const {
  FAISS_ASSERT(listId < deviceListLengths_.size());

  return deviceListLengths_[listId];
}

std::vector<long>
IVFBase::getListIndices(int listId) const {
  FAISS_ASSERT(listId < numLists_);
  FAISS_ASSERT(listId < deviceListIndices_.size());

  if (indicesOptions_ == INDICES_32_BIT) {
    auto intInd = deviceListIndices_[listId]->copyToHost<int>(
      resources_->getDefaultStreamCurrentDevice());

    std::vector<long> out(intInd.size());
    for (size_t i = 0; i < intInd.size(); ++i) {
      out[i] = (long) intInd[i];
    }

    return out;
  } else if (indicesOptions_ == INDICES_64_BIT) {

    return deviceListIndices_[listId]->copyToHost<long>(
      resources_->getDefaultStreamCurrentDevice());
  } else {
    // unhandled indices type (includes INDICES_IVF)
    FAISS_ASSERT(false);
    return std::vector<long>();
  }
}

void
IVFBase::addIndicesFromCpu_(int listId,
                            const long* indices,
                            size_t numVecs) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  auto& listIndices = deviceListIndices_[listId];
  auto prevIndicesData = listIndices->data();

  if (indicesOptions_ == INDICES_32_BIT) {
    // Make sure that all indices are in bounds
    std::vector<int> indices32(numVecs);
    for (size_t i = 0; i < numVecs; ++i) {
      auto ind = indices[i];
      FAISS_ASSERT(ind <= (long) std::numeric_limits<int>::max());
      indices32[i] = (int) ind;
    }

    listIndices->append((unsigned char*) indices32.data(),
                        numVecs * sizeof(int),
                        stream,
                        true /* exact reserved size */);
  } else if (indicesOptions_ == INDICES_64_BIT) {
    listIndices->append((unsigned char*) indices,
                        numVecs * sizeof(long),
                        stream,
                        true /* exact reserved size */);
  } else {
    // indices are not stored
    FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
  }

  if (prevIndicesData != listIndices->data()) {
    deviceListIndexPointers_[listId] = listIndices->data();
  }
}

} } // namespace
