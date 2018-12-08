#include <iostream>
#include <vector>

#include "../gpu/StandardGpuResources.h"
#include "../gpu/utils/HostTensor.cuh"
#include "../gpu/utils/DeviceTensor.cuh"
#include "../gpu/utils/DeviceUtils.h"
#include "../gpu/utils/CopyUtils.cuh"
#include "../gpu/impl/IVFID.cuh"

using namespace std;
using namespace faiss;

gpu::IVFID *newIVFID()
{
    // 0. get device
    int deviceID = gpu::getCurrentDevice();    
    if (deviceID < 0)
    {
        cout << "fail to getCurrentDevice!" << endl;
        return NULL;
    }

    // 1. create gpu resource
    auto resource = new gpu::StandardGpuResources();
    resource->setTempMemory(100<<20);
    resource->initializeForDevice(deviceID);

    // 2. new ivf id
    gpu::IVFID *ivfID = new gpu::IVFID(resource,
                                       gpu::IndicesOptions::INDICES_32_BIT,
                                       gpu::MemorySpace::Device);
    if (NULL == ivfID)
    {
        cout << "fail to new IVFID";
    }
    return ivfID;    
}

void testIVFID()
{
    // 0. new IVFID
    auto idIdx = newIVFID();
    if (NULL == idIdx)
    {
        cout << "new IVFID fail" << endl;
        return;
    }
    idIdx->dump_ids();
    cout << "before insert, dump ids" << endl;

    // 1. insert ids
    int intNum = 777;
    vector<int> ids(intNum); 
    for (int i = 0; i < intNum; ++i)
    {
        ids[i] = i; 
    }
    idIdx->add(ids);
    /*
    int intNum = 16;
    long *ids = new long[intNum];
    for (int i = 0; i < intNum; ++i)
    {
        ids[i] = i;
    }
    idIdx->add_from_cpu(0, ids, intNum);
    */

    // 2. dump ids
    idIdx->dump_ids();

    // 3. delete ids
    //idIdx->testSetTensor();
    for (int i = 0; i < intNum; ++i)
    {
        idIdx->remove_id(i);
    }

    // 4. dump ids

    // 5. insert ids

    // 6. dump ids 
}

void testDeviceTensor()
{
    // 0. get device
    int deviceID = gpu::getCurrentDevice();    
    if (deviceID < 0)
    {
        cout << "fail to getCurrentDevice!" << endl;
        return;
    }

    // 1. create gpu resource
    auto resource = new gpu::StandardGpuResources();
    resource->setTempMemory(100<<20);
    resource->initializeForDevice(deviceID);

    // 2. copy device tensor
    vector<int> ids{1, 2, 3, 4};
    gpu::HostTensor<int, 1, true> intTensor({(int) ids.size()});
    for (int i = 0; i < ids.size(); ++i)
    {
        intTensor[i] = ids[i];
    }

    auto &mem = resource->getMemoryManagerCurrentDevice();
    auto stream = resource->getDefaultStreamCurrentDevice();
    gpu::DeviceTensor<int, 1, true> intDeviceTensor(mem, intTensor, stream);

    int *checkIDs = new int[ids.size()];
    for (int i = 0; i < ids.size(); ++i)
    {
        cout << "i[" << i << "] = " << checkIDs[i] << endl;
    }
    gpu::fromDevice(intDeviceTensor.data(), checkIDs, ids.size(), stream);
    for (int i = 0; i < ids.size(); ++i)
    {
        cout << "i[" << i << "] = " << checkIDs[i] << endl;
    }

    int i;
    cin >> i; 
    cout << "achor" << endl;
}

int main()
{
    testIVFID();
    //testDeviceTensor();
    return 0;
}
