#pragma once

#include "descriptors_cpu.hpp"
#include "descriptors_gpu.hpp"

#include <map>
#include <memory>


struct placed_descriptor_set_t {
    size_t ref_count;
    descriptor_cpu_gpu_handle_t start;
    size_t num;
};

template<size_t N>
struct sampler_heap_t {
public:
    sampler_heap_t(ID3D12Device* device) :
        heap { device }
    { }

public:
    descriptors_gpu_t<D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, N> heap;
    std::map<struct descriptor_set_t *, placed_descriptor_set_t> placed_sets;
};

template<size_t N>
class sampler_heap_cpu_t {
public:
    sampler_heap_cpu_t(ID3D12Device* device) :
        _handle_size { device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) },
        allocator { N }
    {
        const D3D12_DESCRIPTOR_HEAP_DESC desc {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            N,
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            0,
        };
        auto const hr { device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&this->_heap)) };
        // TODO: error

        this->start = this->_heap->GetCPUDescriptorHandleForHeapStart();
    }

    auto alloc(size_t num) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        const auto range = this->allocator.alloc(num);
        if (range) {
            return D3D12_CPU_DESCRIPTOR_HANDLE { this->start.ptr + this->_handle_size * range->_start };
        } else {
            // TODO
            assert(!"Not enough free descriptors in the allocator");
            return D3D12_CPU_DESCRIPTOR_HANDLE { 0 };
        }
    }

    auto assign_set(D3D12_CPU_DESCRIPTOR_HANDLE handle, struct descriptor_set_t* set) {
        this->_offset_to_set.emplace(static_cast<UINT>(handle.ptr - this->start.ptr), set);
    }

    auto get_set(UINT offset) -> struct descriptor_set_t* {
        return this->_offset_to_set[offset];
    }

    auto handle_size() const -> UINT {
        return this->_handle_size;
    }

    auto heap() const -> ID3D12DescriptorHeap* {
        return this->_heap.Get();
    }

private:
    ComPtr<ID3D12DescriptorHeap> _heap;
    D3D12_CPU_DESCRIPTOR_HANDLE start;
    UINT _handle_size;

    free_list allocator;
    std::map<UINT, struct descriptor_set_t *> _offset_to_set;
};

template<size_t N, size_t M>
class sampler_heaps_t {
public:
    sampler_heaps_t(ID3D12Device* device) :
        _gpu_heaps { },
        _cpu_heap { device },
        _device { device }
    { }

    /// Allocate a slice of the descriptor heap.
    ///
    /// Only a slice of the underlying CPU heap!
    auto alloc(size_t num) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        return this->_cpu_heap.alloc(num);
    }

    auto add_gpu_heap() -> heap_index_t {
        this->_gpu_heaps.push_back(
            sampler_heap_t<N>(this->_device)
        );
        return this->_gpu_heaps.size() - 1;
    }

    auto handle_size() const -> UINT {
        return this->_cpu_heap.handle_size();
    }

    auto cpu_heap() const -> ID3D12DescriptorHeap* {
        return this->_cpu_heap.heap();
    }

    auto sampler_heap(heap_index_t index) -> sampler_heap_t<N>& {
        return this->_gpu_heaps[index];
    }

    auto assign_set(D3D12_CPU_DESCRIPTOR_HANDLE handle, struct descriptor_set_t* set) {
        this->_cpu_heap.assign_set(handle, set);
    }

    auto get_set(UINT offset) -> struct descriptor_set_t* {
        return this->_cpu_heap.get_set(offset);
    }

private:
    // Only remove heaps on destruction, we keeping weak indices into this vector.
    std::vector<sampler_heap_t<N>> _gpu_heaps;
    sampler_heap_cpu_t<M> _cpu_heap;

    ID3D12Device* _device;
};
