#include <unordered_map>
#include <iostream>
#include <sstream>

#include <thrust/copy.h>

#include "../GpuResources.h"
#include "../utils/HostTensor.cuh"
#include "../utils/CopyUtils.cuh"

#include "IVFID.cuh"
#include "InvertedListAppend.cuh"


namespace faiss { namespace gpu {

using namespace std; 
using namespace faiss::gpu;

void IVFID::getThrustVector()
{
  std::cout << "------------------------" << std::endl;
  std::cout << "deviceListPointers ---> ";
  thrust::copy(deviceListIndexPointers_.begin(),
               deviceListIndexPointers_.end(),
               std::ostream_iterator<void*>(std::cout, " "));

  std::cout << "\ndeviceListPointers ---> ";
  thrust::copy(deviceListLengths_.begin(),
               deviceListLengths_.end(),
               std::ostream_iterator<int>(std::cout, " "));
  std::cout << "-------------------------" << std::endl;
}

IVFID::IVFID(GpuResources* resources,
               /// We do not own this reference
               IndicesOptions indicesOptions,
               MemorySpace space)
    : IVFBase(resources, indicesOptions, space, NLIST) 
{
}

IVFID::~IVFID()
{
}

void IVFID::add_from_cpu(int listId,
                        const long* indices,
                        size_t numVecs) {
  // This list must already exist
  FAISS_ASSERT(listId < deviceListIndices_.size());
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // If there's nothing to add, then there's nothing we have to do
  if (numVecs == 0) {
    return;
  }

  int preNum = deviceListIndices_[listId]->size() / sizeof(int);

  // Handle the indices as well
  addIndicesFromCpu_(listId, indices, numVecs);

  // And our size has changed too
  int listLength = preNum + numVecs;
  deviceListLengths_[listId] = listLength; 

  // We update this as well, since the multi-pass algorithm uses it
  maxListLength_ = std::max(maxListLength_, listLength);

  // device_vector add is potentially happening on a different stream
  // than our default stream
  if (stream != 0) {
    streamWait({stream}, {0});
  }
}

/*
 * 0. get assign count
 * 1. alloc memory in GPU
 * 2. generate tensor
 * 3. update info
 */
int IVFID::add(const vector<int>& ids)
{
    if (ids.empty()) { return 0; }

    auto& mem = resources_->getMemoryManagerCurrentDevice();
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // vector id -> offset in list
    // (we already have vector id -> list id in listIds)
    HostTensor<int, 1, true> listOffsetHost({ static_cast<int>(ids.size()) });
    HostTensor<int, 1, true> listIdsHost({ static_cast<int>(ids.size()) });

    unordered_map<int, int> assignCounts;
    for (int i = 0; i < ids.size(); ++i)
    {
        int ivfIdx = ids[i] % numLists_;
        int offset = deviceListIndices_[ivfIdx]->size() / sizeof(int);

        auto it = assignCounts.find(ivfIdx);
        if (it != assignCounts.end())
        {
            offset += it->second; 
            ++it->second;
        } else {
            assignCounts[ivfIdx] = 1;
        }
        listIdsHost[i] = ivfIdx;
        listOffsetHost[i] = offset;
    }

    {
        ostringstream strIds;
        ostringstream strOffset;
        for (int i = 0; i < ids.size(); ++i)
        {
            strIds << listIdsHost[i] << " "; 
            strOffset << listOffsetHost[i] << " ";
        }
        cout << "listIds, size:" << listIdsHost.getSize(0) << ":" << strIds.str() << endl;
        cout << "listOffsetHost, size:" << listOffsetHost.getSize(0) << ":" << strOffset.str() << endl;
    }

    {
        cout << "before resize, maxListLength_:" << maxListLength_ << endl;
        for (int i = 0; i < NLIST; ++i)
        {
            auto &indices = deviceListIndices_[i];
            cout << "indices[" << i << "]'s size:" << indices->size() << " cap:" << indices->capacity() << endl;
        }
        // resize device vector
        for (const auto &count : assignCounts) 
        {
            auto &indices = deviceListIndices_[count.first];
            int newSize = count.second;
            if (indicesOptions_ == INDICES_32_BIT) {
                size_t indexSize = sizeof(int);
                newSize += indices->size() / indexSize;
                indices->resize(indices->size() + indexSize * count.second, stream);
            } else {
                FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
            }
            maxListLength_ = std::max(maxListLength_, newSize);
        }
        cout << "after resize, maxListLength_:" << maxListLength_ << endl;
        for (int i = 0; i < NLIST; ++i)
        {
            auto &indices = deviceListIndices_[i];
            cout << "indices[" << i << "]'s size:" << indices->size() << " cap:" << indices->capacity() << endl;
        }

        // Update all pointers and sizes on the device for lists that we appended to
        vector<int> listIds(assignCounts.size());
        int i = 0;
        for (auto& count: assignCounts)
        {
            cout << "count.first:" << count.first << ", count.second:" << count.second << endl;
            listIds[i++] = count.first;
        }
        updateDeviceListInfo_(listIds, stream);
    }

    {
        // generate indices
        vector<long> longIdx(ids.begin(), ids.end());
        auto indices = toDevice<long, 1>(resources_,
                              0,  // gpu device idx, default 0
                              const_cast<long*>(longIdx.data()),
                              stream,
                              {(int) ids.size()});

        // host data struct to device struct
        DeviceTensor<int, 1, true> listIds(mem, listIdsHost, stream);
        DeviceTensor<int, 1, true> listOffset(mem, listOffsetHost, stream);

        cout << "listIds size:" << listIds.getSize(0)
             << ", listOffset size:" << listOffset.getSize(0)
             << ", indices size:" << indices.getSize(0) << endl;

        // Now, for each list to which a vector is being assigned, write it
        // listIDs change to device tensor
        runIVFIDInvertedListAppend(listIds,
                                   listOffset,
                                   indices,
                                   deviceListIndexPointers_,
                                   indicesOptions_,
                                   stream);
        auto checkList = getListIndices(0);
    }
}

int IVFID::remove_id(int id)
{
    auto stream = resources_->getDefaultStreamCurrentDevice();

    DeviceTensor<int, 1, true> offset({NLIST});
    // TODO
    runIVFIDInvertedListFind((long)id,
                             deviceListIndexPointers_,
                             deviceListLengths_,
                             indicesOptions_,
                             offset,
                             stream);
    int *hostOffset = new int[NLIST];
    fromDevice(offset, hostOffset, stream);
    ostringstream oss;
    oss << "offset:";
    for (int i = 0; i < NLIST; ++i)
    {
        oss << " " << hostOffset[i];
    }
    cout << oss.str() << endl;
    delete []hostOffset;
    return 0;
}

/*
 * delete one id from gpu per time.
 * 0. delete from gpu
 * 1. update info (DeviceVector, IVFID)
 */
int IVFID::remove_ids(const std::vector<int>& ids)
{
    // TODO
    return 0;
}

void IVFID::dump_ids()
{
    for (int i = 0; i < NLIST; ++i)
    {
        auto indice = getListIndices(i);
        ostringstream oss;
        oss << "list[" << i << "]: ";
        for (const auto &id : indice)
        {
            oss << id << " "; 
        }
        cout << oss.str() << endl; 
    }
}

} }