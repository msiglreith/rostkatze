#include "icd.hpp"

#include <d3d12.h>
#include <wrl.h>

#include <list>
#include <optional>

#include <stdx/range.hpp>

using namespace stdx;

// TODO: multi-threading
class free_list {
public:
    free_list(size_t size) :
        _size { size }
    {
        this->_list.push_back(range(_size));
    }

    auto alloc(size_t num) -> std::optional<range_t<size_t>> {
        if (!num) {
            return std::optional<range_t<size_t>>(range_t<size_t>(0u, 0u));
        }

        for (auto it = this->_list.begin(); it != this->_list.end(); ++it) {
            if (it->_end >= it->_start + num) {
                std::list<range_t<size_t>> tail;
                tail.splice(tail.begin(), this->_list, it, this->_list.end());

                range_t<size_t> node { tail.front() };
                const range_t<size_t> allocated { node._start, node._start + num };
                node._start += num;

                if (node._start < node._end) {
                    this->_list.push_back(node);
                }
                tail.pop_front();
                this->_list.splice(this->_list.end(), tail);

                return allocated;
            }
        }

        return std::nullopt;
    }

private:
    size_t _size;
    std::list<range_t<size_t>> _list;
};

using descriptor_cpu_gpu_handle_t = std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE>;

/// GPU descriptor heap
template<D3D12_DESCRIPTOR_HEAP_TYPE Ty, size_t N>
class descriptors_gpu_t {
public:
    descriptors_gpu_t(ID3D12Device* device) :
        _handle_size { device->GetDescriptorHandleIncrementSize(Ty) },
        allocator { N }
    {
        const D3D12_DESCRIPTOR_HEAP_DESC desc { Ty, N, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
        auto const hr { device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&this->_heap)) };
        // TODO: error

        this->start_cpu = this->_heap->GetCPUDescriptorHandleForHeapStart();
        this->start_gpu = this->_heap->GetGPUDescriptorHandleForHeapStart();
    }

    /// Allocate a slice of the descriptor heap.
    auto alloc(size_t num) -> descriptor_cpu_gpu_handle_t {
        const auto range = this->allocator.alloc(num);
        if (range) {
            return std::make_tuple(
                D3D12_CPU_DESCRIPTOR_HANDLE { this->start_cpu.ptr + this->_handle_size * range->_start },
                D3D12_GPU_DESCRIPTOR_HANDLE { this->start_gpu.ptr + this->_handle_size * range->_start }
            );
        } else {
            // TODO
            assert(!"Not enough free descriptors in the allocator");
            return std::make_tuple(
                D3D12_CPU_DESCRIPTOR_HANDLE { 0 },
                D3D12_GPU_DESCRIPTOR_HANDLE { 0 }
            );
        }
    }

    /// Free a descriptor heap slice.
    auto free(descriptor_cpu_gpu_handle_t handle, size_t num) -> void {
        WARN("GPU descriptor free unimplemented");
    }

    auto handle_size() const -> UINT {
        return this->_handle_size;
    }

    auto heap() const -> ID3D12DescriptorHeap* {
        return this->_heap.Get();
    }

private:
    ComPtr<ID3D12DescriptorHeap> _heap;
    D3D12_CPU_DESCRIPTOR_HANDLE start_cpu;
    D3D12_GPU_DESCRIPTOR_HANDLE start_gpu;
    UINT _handle_size;

    free_list allocator;
};
