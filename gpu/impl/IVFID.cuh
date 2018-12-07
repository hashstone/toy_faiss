#pragma once

#include "IVFBase.cuh"

namespace faiss { namespace gpu {

class IVFID : public IVFBase {
    static const int NLIST = 32;

public:
    IVFID(GpuResources* resources,
           /// We do not own this reference
           IndicesOptions indicesOptions,
           MemorySpace space);

    ~IVFID() override;

    int add(const std::vector<int>& ids);

    void add_from_cpu(int listId, const long* indices, size_t numVecs);

    int remove_id(int id);

    int remove_ids(const std::vector<int>& ids);

    void getThrustVector();

    void dump_ids();
};

} }