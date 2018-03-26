#pragma once

#include "descriptors_gpu.hpp"

template<D3D12_DESCRIPTOR_HEAP_TYPE Ty, size_t N>
class descriptors_virtual_t {
public:
    descriptors_virtual_t(ID3D12Device* device) :
        heap { device }
    { }

private:
    descriptors_gpu_t<Ty, N> heap;
};
