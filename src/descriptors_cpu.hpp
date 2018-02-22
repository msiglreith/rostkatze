
#include "icd.hpp"

#include <d3d12.h>
#include <wrl.h>

#include <bitset>
#include <optional>

using namespace Microsoft::WRL;

/// A CPU descriptor heap.
template<size_t N>
class heap_cpu_t {
public:
    heap_cpu_t(ComPtr<ID3D12DescriptorHeap> heap, UINT handle_size) :
        heap { heap },
        occupacy { 0 },
        start { heap->GetCPUDescriptorHandleForHeapStart() },
        handle_size { handle_size }
    { }

    auto alloc() -> D3D12_CPU_DESCRIPTOR_HANDLE {
        if (full()) {
            assert("Heap is full");
        }

        // TODO: possible optimizations via bitscans..
        auto slot { 0 };
        for (; slot < N; ++slot) {
            if (!this->occupacy[slot]) {
                this->occupacy.set(slot);
                break;
            }
        }

        return D3D12_CPU_DESCRIPTOR_HANDLE {
            this->start.ptr + this->handle_size * slot
        };
    }

    auto free(D3D12_CPU_DESCRIPTOR_HANDLE handle) -> void {
        this->occupacy.reset((handle.ptr - this->start.ptr) / this->handle_size);
    }

    auto full() const -> bool {
        return this->occupacy.all();
    }

private:
    ComPtr<ID3D12DescriptorHeap> heap;
    D3D12_CPU_DESCRIPTOR_HANDLE start;
    UINT handle_size;

    std::bitset<N> occupacy;
};

using heap_index_t = size_t;

// CPU descriptor heap manager.
// TODO: multi-thread access
template<D3D12_DESCRIPTOR_HEAP_TYPE Ty, size_t N>
class descriptors_cpu_t {
public:
    descriptors_cpu_t(ID3D12Device* device) :
        device { device },
        heaps { },
        free_list { },
        handle_size { device->GetDescriptorHandleIncrementSize(Ty) }
    { }

    auto alloc() -> std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> {
        do {
            if (this->free_list.empty()) {
                ComPtr<ID3D12DescriptorHeap> heap { nullptr };
                const D3D12_DESCRIPTOR_HEAP_DESC desc { Ty, N, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
                this->device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap));

                this->heaps.emplace_back(heap, this->handle_size);
                this->free_list.push_back(this->heaps.size()-1);
            }

            const auto free_heap { free_list.back() };
            if (this->heaps[free_heap].full()) {
                free_list.pop_back();
            } else {
                return std::make_tuple(this->heaps[free_heap].alloc(), free_heap);
            }
        } while (true);
    }

    auto free(D3D12_CPU_DESCRIPTOR_HANDLE handle, heap_index_t index) {
        this->heaps[index].free(handle);
    }
private:
    ID3D12Device* device;
    std::vector<heap_cpu_t<N>> heaps;
    std::vector<heap_index_t> free_list;
    UINT handle_size;
};

