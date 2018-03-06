
// Copyright 2017 The Gfx-rs Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2017 The NXT Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "icd.hpp"
#include "descriptors_cpu.hpp"
#include "descriptors_gpu.hpp"

#include <vk_icd.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include <gsl/gsl>
#include <stdx/hash.hpp>
#include <stdx/match.hpp>
#include <stdx/range.hpp>

#include <spirv-cross/spirv_hlsl.hpp>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <wrl.h>

#include <d3d12.h>
#include <d3dx12.h>
#include <dxgi1_6.h>
#include <D3Dcompiler.h>

using namespace gsl;
using namespace stdx;
using namespace Microsoft::WRL;

auto saturated_add(uint64_t x, uint64_t y) -> uint64_t {
    const auto result { x + y };
    if (result < x) {
        return ~0u;
    } else {
        return result;
    }
}

auto init_debug_interface() {
    static auto initialized { false };
    if (initialized) {
        return;
    }

    initialized = true;
    ComPtr<ID3D12Debug> debug_controller {};
    auto hr = ::D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller));
    if (SUCCEEDED(hr)) {
        debug_controller->EnableDebugLayer();
    }
}

static const size_t MAX_VERTEX_BUFFER_SLOTS = 16;

static const size_t PUSH_CONSTANT_REGISTER_SPACE = 0;
static const size_t DESCRIPTOR_TABLE_INITIAL_SPACE = 1;

static const char* copy_buffer_to_image_cs = R"(
ByteAddressBuffer src : register(t0);
RWTexture2D<uint> dst : register(u1);

struct Region {
    uint buffer_offset;
    uint buffer_row_length;
    uint buffer_slice_pitch;
};
ConstantBuffer<Region> region : register(b0);

[numthreads( 1, 1, 1 )]
void CopyBufferToImage(uint3 dispatch_thread_id : SV_DispatchThreadID) {
    dst[dispatch_thread_id.xy] =
        src.Load(4 * dispatch_thread_id.x + region.buffer_offset + region.buffer_row_length * dispatch_thread_id.y);
}
)";


// TODO: Very basic implementation, doesnt handle offset correctly
static const char* blit_2d_cs = R"(
Texture2DArray src : register(t0);
SamplerState src_sampler : register(s0);
RWTexture2DArray<float4> dst : register(u1);

struct Region {
    int src_offset_x;
    int src_offset_y;
    int src_offset_z;

    uint src_extent_x;
    uint src_extent_y;
    uint src_extent_z;

    uint dst_offset_x;
    uint dst_offset_y;
    uint dst_offset_z;

    uint dst_extent_x;
    uint dst_extent_y;
    uint dst_extent_z;

    uint src_size_x;
    uint src_size_y;
    uint src_size_z;
};
ConstantBuffer<Region> region : register(b0);

[numthreads( 1, 1, 1 )]
void BlitImage2D(uint3 dispatch_thread_id : SV_DispatchThreadID) {
    uint3 dst_offset = uint3(region.dst_offset_x, region.dst_offset_y, region.dst_offset_z);
    int3 src_offset = int3(region.src_offset_x, region.src_offset_y, region.src_offset_z);

    float u_offset = float(dispatch_thread_id.x) + 0.5;
    float v_offset = float(dispatch_thread_id.y) + 0.5;
    float w_offset = float(dispatch_thread_id.z) + 0.5;

    float scale_u = float(region.src_extent_x) / float(region.dst_extent_x);
    float scale_v = float(region.src_extent_y) / float(region.dst_extent_y);
    float scale_w = float(region.src_extent_z) / float(region.dst_extent_z);

    float3 uvw = float3(
        u_offset * scale_u / float(region.src_size_x),
        v_offset * scale_v / float(region.src_size_y),
        w_offset * scale_w / float(region.src_size_z)
    );

    dst[dispatch_thread_id + dst_offset] = src.SampleLevel(src_sampler, uvw, src_offset);
}
)";

enum queue_family {
    QUEUE_FAMILY_GENERAL_PRESENT = 0,
    QUEUE_FAMILY_GENERAL,
    QUEUE_FAMILY_COMPUTE,
    QUEUE_FAMILY_COPY,
};

struct heap_properties_t {
    D3D12_CPU_PAGE_PROPERTY page_property;
    D3D12_MEMORY_POOL memory_pool;
};

// https://msdn.microsoft.com/de-de/library/windows/desktop/dn788678(v=vs.85).aspx
static const std::array<heap_properties_t, 3> HEAPS_NUMA = {{
    // DEFAULT
    { D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, D3D12_MEMORY_POOL_L1 },
    // UPLOAD
    { D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0  },
    // READBACK
    { D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0 }
}};

static const std::array<heap_properties_t, 3> HEAPS_UMA = {{
    // DEFAULT
    { D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, D3D12_MEMORY_POOL_L0 },
    // UPLOAD
    { D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0 },
    // READBACK
    { D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0 }
}};

static const std::array<heap_properties_t, 3> HEAPS_CCUMA = {{
    // DEFAULT
    { D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, D3D12_MEMORY_POOL_L0 },
    // UPLOAD
    { D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0 },
    //READBACK
    { D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0 }
}};

auto compare_op(VkCompareOp op) {
    switch (op) {
        case VK_COMPARE_OP_NEVER: return D3D12_COMPARISON_FUNC_NEVER;
        case VK_COMPARE_OP_LESS: return D3D12_COMPARISON_FUNC_LESS;
        case VK_COMPARE_OP_EQUAL: return D3D12_COMPARISON_FUNC_EQUAL;
        case VK_COMPARE_OP_LESS_OR_EQUAL: return D3D12_COMPARISON_FUNC_LESS_EQUAL;
        case VK_COMPARE_OP_GREATER: return D3D12_COMPARISON_FUNC_GREATER;
        case VK_COMPARE_OP_NOT_EQUAL: return D3D12_COMPARISON_FUNC_NOT_EQUAL;
        case VK_COMPARE_OP_GREATER_OR_EQUAL: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        case VK_COMPARE_OP_ALWAYS: return D3D12_COMPARISON_FUNC_ALWAYS;
        default: return D3D12_COMPARISON_FUNC_ALWAYS;
    }
};

auto up_align(UINT v, UINT alignment) -> UINT {
    return (v + alignment - 1) & ~(alignment - 1);
}

// Forward declerations
class instance_t;
class device_t;
class queue_t;

struct physical_device_t;
struct semaphore_t;
struct surface_t;
struct device_memory_t;
struct image_t;
struct render_pass_t;
struct framebuffer_t;
struct image_view_t;

struct physical_device_properties_t {
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t vendor_id;
    uint32_t device_id;
    VkPhysicalDeviceType device_type;
    std::array<char, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE> device_name;
};

struct physical_device_t {
    std::mutex is_open;
    ComPtr<IDXGIAdapter4> adapter;

    physical_device_properties_t properties;
    std::optional<VkPhysicalDeviceConservativeRasterizationPropertiesEXT> conservative_properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    span<const heap_properties_t> heap_properties;

    VkPhysicalDeviceLimits limits;
};

class instance_t final {
public:
    instance_t() {
        init_logging();
        init_debug_interface();

        DEBUG("creating instance");

        // Create dxgi factory
        {
            IDXGIFactory5 *factory;
            auto hr { ::CreateDXGIFactory2(
                DXGI_CREATE_FACTORY_DEBUG,
                IID_PPV_ARGS(&factory)
            )};

            this->dxgi_factory = factory;
        }

        struct adapter_info_t {
            ComPtr<IDXGIAdapter4> adapter;
            physical_device_properties_t properties;
            std::optional<VkPhysicalDeviceConservativeRasterizationPropertiesEXT> conservative_properties;
            VkPhysicalDeviceMemoryProperties memory_properties;
            span<const heap_properties_t> heap_properties;
            VkPhysicalDeviceLimits limits;
        };

        //
        std::vector<adapter_info_t> adapters;
        auto i { 0 };
        while (true) {
            // Query adapter
            ComPtr<IDXGIAdapter1> dxgi_adapter {};
            if (this->dxgi_factory->EnumAdapters1(i, &dxgi_adapter) == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            i += 1;

            ComPtr<IDXGIAdapter4> adapter;
            if (FAILED(dxgi_adapter.As<IDXGIAdapter4>(&adapter))) {
                ERR("Couldn't convert adapter to `IDXGIAdapter4`!");
                continue;
            }

            DXGI_ADAPTER_DESC3 adapter_desc;
            adapter->GetDesc3(&adapter_desc);

            physical_device_properties_t properties {
                VK_MAKE_VERSION(2, 0, 68), // TODO: 2.0 for debugging?
                0,
                adapter_desc.VendorId,
                adapter_desc.DeviceId,
                VK_PHYSICAL_DEVICE_TYPE_OTHER, // TODO
            };
            static_assert(VK_MAX_PHYSICAL_DEVICE_NAME_SIZE >= sizeof(WCHAR) * 128);
            WideCharToMultiByte(
                CP_UTF8,
                WC_NO_BEST_FIT_CHARS,
                adapter_desc.Description,
                128,
                properties.device_name.data(),
                VK_MAX_PHYSICAL_DEVICE_NAME_SIZE,
                nullptr,
                nullptr
            );

            // temporary device
            ComPtr<ID3D12Device3> device;
            {
                const auto hr { ::D3D12CreateDevice(
                    adapter.Get(),
                    D3D_FEATURE_LEVEL_11_0,
                    IID_PPV_ARGS(&device)
                )};

                if (FAILED(hr)) {
                    // Doesn't support dx12
                    continue;
                }
            }

            D3D12_FEATURE_DATA_D3D12_OPTIONS feature_options { 0 };
            {
                const auto hr {
                    device->CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS,
                        &feature_options,
                        sizeof(feature_options)
                    )
                };
                // TODO: error handling
            }
            std::optional<VkPhysicalDeviceConservativeRasterizationPropertiesEXT> conservative_properties { std::nullopt };
            if (feature_options.ConservativeRasterizationTier) {
                VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative;
                conservative.maxExtraPrimitiveOverestimationSize = 0.0;
                conservative.conservativePointAndLineRasterization = false;
                conservative.degenerateLinesRasterized = false; // not supported anyways?
                conservative.conservativeRasterizationPostDepthCoverage = false; // TODO: check again

                switch (feature_options.ConservativeRasterizationTier) {
                    case D3D12_CONSERVATIVE_RASTERIZATION_TIER_1: {
                        conservative.primitiveOverestimationSize = 0.5;
                        conservative.primitiveUnderestimation = false;
                        conservative.degenerateTrianglesRasterized = false;
                        conservative.fullyCoveredFragmentShaderInputVariable = false;

                    } break;
                    case D3D12_CONSERVATIVE_RASTERIZATION_TIER_2: {
                        conservative.primitiveOverestimationSize = 1.0 / 256.0f;
                        conservative.primitiveUnderestimation = false;
                        conservative.degenerateTrianglesRasterized = true;
                        conservative.fullyCoveredFragmentShaderInputVariable = false;
                    } break;
                    case D3D12_CONSERVATIVE_RASTERIZATION_TIER_3: {
                        conservative.primitiveOverestimationSize = 1.0 / 256.0f;
                        conservative.primitiveUnderestimation = true;
                        conservative.degenerateTrianglesRasterized = true;
                        conservative.fullyCoveredFragmentShaderInputVariable = true; // TODO: SPIRV-Cross support
                    } break;
                }
                conservative_properties = conservative;
            }


            D3D12_FEATURE_DATA_ARCHITECTURE feature_architecture { 0 };
            {
                const auto hr {
                    device->CheckFeatureSupport(
                        D3D12_FEATURE_ARCHITECTURE,
                        &feature_architecture,
                        sizeof(feature_architecture)
                    )
                };
                if (FAILED(hr)) {
                    ERR("Couldn't query feature architcture data");
                    // TODO
                }
            }
            const auto uma { feature_architecture.UMA };
            const auto cc_uma { feature_architecture.CacheCoherentUMA };

            VkPhysicalDeviceMemoryProperties memory_properties;

            span<const heap_properties_t> heap_properties;

            // Memory types
            // TODO: HEAP TIER 1
            memory_properties.memoryTypeCount = 3;
            if (uma && cc_uma) {
                heap_properties = span<const heap_properties_t>{ HEAPS_CCUMA };

                // Default
                memory_properties.memoryTypes[0] = {
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    0
                };
                // Upload
                memory_properties.memoryTypes[1] = {
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    0
                };
                // Readback
                memory_properties.memoryTypes[2] = {
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    0
                };
            } else if (uma && !cc_uma) {
                heap_properties = span<const heap_properties_t>{ HEAPS_UMA };

                // Default
                memory_properties.memoryTypes[0] = {
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    0
                };
                // Upload
                memory_properties.memoryTypes[1] = {
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    0
                };
                // Readback
                memory_properties.memoryTypes[2] = {
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    0
                };
            } else { // NUMA
                heap_properties = span<const heap_properties_t>{ HEAPS_NUMA };

                // Default
                memory_properties.memoryTypes[0] = {
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    0
                };
                // Upload
                memory_properties.memoryTypes[1] = {
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    1
                };
                // Readback
                memory_properties.memoryTypes[2] = {
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    1
                };
            }

            // Memory heaps
            auto query_memory = [&] (DXGI_MEMORY_SEGMENT_GROUP segment) {
                DXGI_QUERY_VIDEO_MEMORY_INFO memory_info;
                const auto hr { adapter->QueryVideoMemoryInfo(0, segment, &memory_info) };
                // TODO: error handling
                return memory_info.Budget;
            };

            const auto memory_local_size { query_memory(DXGI_MEMORY_SEGMENT_GROUP_LOCAL) };
            memory_properties.memoryHeaps[0] = { memory_local_size, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT };

            if (!uma) {
                memory_properties.memoryHeapCount = 2;
                memory_properties.memoryHeaps[1] = { query_memory(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL), 0 };
            } else {
                memory_properties.memoryHeapCount = 1;
            }

            // Limits
            VkPhysicalDeviceLimits limits { 0 };
            limits.maxImageDimension1D = D3D12_REQ_TEXTURE1D_U_DIMENSION;
            limits.maxImageDimension2D = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
            limits.maxImageDimension3D = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
            limits.maxImageDimensionCube = D3D12_REQ_TEXTURECUBE_DIMENSION;
            // TODO: missing fields
            limits.maxPushConstantsSize = 4 * D3D12_MAX_ROOT_COST;
            // TODO: missing fields
            limits.maxComputeSharedMemorySize = D3D12_CS_THREAD_LOCAL_TEMP_REGISTER_POOL;
            // TODO: missing fields
            limits.maxSamplerAnisotropy = 16;
            // TODO: missing fields
            limits.framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT; // TODO
            limits.framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT; // TODO
            limits.framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT; // TODO
            limits.framebufferNoAttachmentsSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT; // TODO

            adapters.emplace_back(
                adapter_info_t {
                    adapter,
                    properties,
                    conservative_properties,
                    memory_properties,
                    heap_properties,
                    limits,
                }
            );
        }

        const auto num_adapters { adapters.size() };
        this->_adapters = std::vector<physical_device_t>(num_adapters);
        for (auto i : range(num_adapters)) {
            auto& adapter { this->_adapters[i] };
            adapter.adapter = adapters[i].adapter;
            adapter.properties = adapters[i].properties;
            adapter.conservative_properties = adapters[i].conservative_properties;
            adapter.memory_properties = adapters[i].memory_properties;
            adapter.heap_properties = adapters[i].heap_properties;
            adapter.limits = adapters[i].limits;
        }
    }

    ~instance_t() {

    }

    auto adapters() const -> span<const physical_device_t> {
        return make_span(this->_adapters);
    }

public:
    ComPtr<IDXGIFactory5> dxgi_factory;
    std::vector<physical_device_t> _adapters;
};

class queue_t final {
public:
    queue_t() : queue_t { nullptr, nullptr } { }
    queue_t(queue_t && tmp) :
        loader_magic { ICD_LOADER_MAGIC },
        queue { tmp.queue },
        idle_fence { tmp.idle_fence },
        idle_event { tmp.idle_event }
    {
        tmp.queue = nullptr;
        tmp.idle_fence = nullptr;
        tmp.idle_event = 0;
    }

    queue_t(queue_t const&) = delete;

    queue_t(ComPtr<ID3D12CommandQueue> queue, ComPtr<ID3D12Fence> fence) :
        loader_magic { ICD_LOADER_MAGIC },
        queue { queue },
        idle_fence { fence },
        idle_event { ::CreateEvent(nullptr, TRUE, FALSE, nullptr) } // Event with manual reset
    { }

    ~queue_t() {
        if (this->idle_event) {
            ::CloseHandle(this->idle_event);
        }
    }

    auto operator= (queue_t const&) -> queue_t& = delete;
    auto operator= (queue_t && tmp) -> queue_t& {
        this->queue = std::move(tmp.queue);
        this->idle_fence = std::move(tmp.idle_fence);
        this->idle_event = std::move(tmp.idle_event);

        tmp.queue = nullptr;
        tmp.idle_fence = nullptr;
        tmp.idle_event = 0;

        return *this;
    }

    auto operator->() -> ID3D12CommandQueue* {
        return this->queue.Get();
    }

private:
    /// Dispatchable object
    uintptr_t loader_magic;

public:
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> idle_fence;
    HANDLE idle_event;
};

struct command_pool_t {
    D3D12_COMMAND_LIST_TYPE type;
    ComPtr<ID3D12CommandAllocator> allocator;
};

struct image_t {
    ComPtr<ID3D12Resource> resource;
    D3D12_RESOURCE_ALLOCATION_INFO allocation_info;
    D3D12_RESOURCE_DESC resource_desc;
    format_block_t block_data;
    VkImageUsageFlags usage;
};

// TODO: the whole structure could be stripped a bit
//       and maybe precompute some more information on creation.
struct render_pass_t {
    struct subpass_t {
        std::vector<VkAttachmentReference> input_attachments;
        std::vector<VkAttachmentReference> color_attachments;
        std::vector<VkAttachmentReference> resolve_attachments;
        VkAttachmentReference depth_attachment;
        std::vector<uint32_t> preserve_attachments;
    };

    struct attachment_t {
        VkAttachmentDescription desc;
        std::optional<size_t> first_use;
    };

    std::vector<subpass_t> subpasses;
    std::vector<attachment_t> attachments;
    std::vector<VkSubpassDependency> dependencies;
};

struct framebuffer_t {
    std::vector<image_view_t *> attachments;
};

struct image_view_t {
    ID3D12Resource* image;
    std::optional<std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t>> rtv;
    std::optional<std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t>> dsv;
    std::optional<std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t>> srv;
    std::optional<std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t>> uav; // TODO: destroy
};

struct shader_module_t {
    // Raw SPIR-V code
    std::vector<uint32_t> spirv;
};

enum dynamic_state_flags {
    DYNAMIC_STATE_DEPTH_BIAS = 0x1,
    DYNAMIC_STATE_STENCIL_COMPARE_MASK = 0x2,
    DYNAMIC_STATE_STENCIL_WRITE_MASK = 0x4,
    DYNAMIC_STATE_PRIMITIVE_RESTART = 0x8,
};
using dynamic_states_t = uint32_t;

struct dynamic_state_t {
    dynamic_state_t() :
        depth_bias { 0 },
        depth_bias_clamp { 0.0f },
        depth_bias_slope { 0.0f },
        stencil_read_mask { 0 },
        stencil_write_mask { 0 },
        strip_cut { D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED }
    { }

    INT depth_bias;
    FLOAT depth_bias_clamp;
    FLOAT depth_bias_slope;
    UINT8 stencil_read_mask;
    UINT8 stencil_write_mask;
    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE strip_cut;

    auto operator== (dynamic_state_t const& rhs) const -> bool {
        return
            this->depth_bias == rhs.depth_bias &&
            this->depth_bias_clamp == rhs.depth_bias_clamp &&
            this->depth_bias_slope == rhs.depth_bias_slope &&
            this->stencil_read_mask == rhs.stencil_read_mask &&
            this->stencil_write_mask == rhs.stencil_write_mask &&
            this->strip_cut == rhs.strip_cut;
    }
};

namespace std {
    template<>
    struct hash<dynamic_state_t> {
        auto operator()(dynamic_state_t const& v) const -> std::size_t {
            std::size_t hash { 0 };
            stdx::hash_combine(
                hash,
                v.depth_bias,
                v.depth_bias_clamp,
                v.depth_bias_slope,
                v.stencil_read_mask,
                v.stencil_write_mask,
                v.strip_cut
            );
            return hash;
        }
   };
}

struct blend_factors_t {
    FLOAT factors[4];
};

enum class draw_type {
    DRAW,
    DRAW_INDEXED,
};

struct pipeline_t {
    pipeline_t() :
        signature { nullptr },
        num_signature_entries { 0 },
        num_root_constants { 0 },
        dynamic_states { 0 },
        static_viewports { std::nullopt },
        static_scissors { std::nullopt },
        static_blend_factors { std::nullopt },
        static_depth_bounds { std::nullopt },
        static_stencil_reference { std::nullopt }
    {}

    // There is only a single pipeline, no need for dynamic creation
    // Compute pipelines and graphics pipeline without primitive restart and dynamic states
    struct unique_pso_t {
        ComPtr<ID3D12PipelineState> pipeline;
    };

    // Pipeline with dynamic states
    //  - Viewport and scissor are dynamic per se
    //  - Line width must be 1.0
    //  - Blend constants are dynamic per se
    //  - Depth bounds are dynamic per se (if supported)
    //  - Stencil ref is dynamic per se
    //  - Index primitive restart value must be set dynamically
    //
    // Access to these pipelines need to be ensured by `pso_access`.
    struct dynamic_pso_t {
        std::unordered_map<dynamic_state_t, ComPtr<ID3D12PipelineState>> pipelines;

        // desc for dynamic pipeline creation
        ComPtr<ID3DBlob> vertex_shader;
        ComPtr<ID3DBlob> domain_shader;
        ComPtr<ID3DBlob> hull_shader;
        ComPtr<ID3DBlob> geometry_shader;
        ComPtr<ID3DBlob> pixel_shader;
        D3D12_BLEND_DESC blend_state;
        UINT sample_mask;
        D3D12_RASTERIZER_DESC rasterizer_state;
        D3D12_DEPTH_STENCIL_DESC depth_stencil_state;
        std::vector<D3D12_INPUT_ELEMENT_DESC> input_elements;
        UINT num_render_targets;
        DXGI_FORMAT rtv_formats[8];
        DXGI_FORMAT dsv_format;
        DXGI_SAMPLE_DESC sample_desc;
        D3D12_PRIMITIVE_TOPOLOGY_TYPE topology_type;
    };

    // Shared by compute and graphics
    std::variant<unique_pso_t, dynamic_pso_t> pso;
    // Required for accessing
    std::shared_mutex pso_access;
    ID3D12RootSignature* signature;

    size_t num_signature_entries;
    size_t num_root_constants; // Number of root constants (32bit) in the root signature
    std::vector<VkPushConstantRange> root_constants;

    // Graphics only
    D3D12_PRIMITIVE_TOPOLOGY topology;
    std::array<uint32_t, MAX_VERTEX_BUFFER_SLOTS> vertex_strides;

    bool primitive_restart;
    dynamic_states_t dynamic_states;

    dynamic_state_t static_state;
    std::optional<std::vector<D3D12_VIEWPORT>> static_viewports;
    std::optional<std::vector<D3D12_RECT>> static_scissors;
    std::optional<blend_factors_t> static_blend_factors;
    std::optional<std::tuple<FLOAT, FLOAT>> static_depth_bounds;
    std::optional<UINT> static_stencil_reference;
};

class device_t final {
public:
    device_t(ComPtr<ID3D12Device3> device) :
        loader_magic { ICD_LOADER_MAGIC },
        descriptors_cpu_rtv { device.Get() },
        descriptors_cpu_dsv { device.Get() },
        descriptors_cpu_cbv_srv_uav { device.Get() },
        descriptors_cpu_sampler { device.Get() },
        descriptors_gpu_cbv_srv_uav { device.Get() },
        descriptors_gpu_sampler { device.Get() },
        pso_buffer_to_image { nullptr },
        signature_buffer_to_image { nullptr },
        device { device }
    { }

    ~device_t() {}

    auto operator->() -> ID3D12Device3* {
        return this->device.Get();
    }

    auto create_render_target_view(
        image_t* image,
        VkImageViewType type,
        DXGI_FORMAT format,
        VkImageSubresourceRange const& range
    ) -> std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> {
        auto resource { image->resource.Get() };

        const auto layer_count {
            range.layerCount == VK_REMAINING_ARRAY_LAYERS ?
                image->resource_desc.DepthOrArraySize - range.baseArrayLayer :
                range.layerCount
        };
        // TODO: multisampling

        auto handle { this->descriptors_cpu_rtv.alloc() };

        D3D12_RENDER_TARGET_VIEW_DESC desc { format };
        if (type == VK_IMAGE_TYPE_2D && image->resource_desc.SampleDesc.Count != VK_SAMPLE_COUNT_1_BIT) {
            desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            desc.Texture2DMS = D3D12_TEX2DMS_RTV { };
        } else if (type == VK_IMAGE_VIEW_TYPE_1D && range.baseArrayLayer == 0) {
            desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1D;
            desc.Texture1D = D3D12_TEX1D_RTV { range.baseMipLevel };
        } else if (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY) {
            desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
            desc.Texture1DArray = D3D12_TEX1D_ARRAY_RTV {
                range.baseMipLevel,
                range.baseArrayLayer,
                layer_count
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_2D && range.baseArrayLayer == 0) {
            desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            desc.Texture2D = D3D12_TEX2D_RTV { range.baseMipLevel, 0 };
        } else if (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_2D_ARRAY ||
                   type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
        {
            desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            desc.Texture2DArray = D3D12_TEX2D_ARRAY_RTV {
                range.baseMipLevel,
                range.baseArrayLayer,
                layer_count,
                0
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_3D) {
            desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
            desc.Texture3D = D3D12_TEX3D_RTV { range.baseMipLevel, 0, ~0u };
        }

        this->device->CreateRenderTargetView(resource, &desc, std::get<0>(handle));
        return handle;
    }

    auto create_depth_stencil_view(
        image_t* image,
        VkImageViewType type,
        DXGI_FORMAT format,
        VkImageSubresourceRange const& range
    ) -> std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> {
        auto resource { image->resource.Get() };

        const auto layer_count {
            range.layerCount == VK_REMAINING_ARRAY_LAYERS ?
                image->resource_desc.DepthOrArraySize - range.baseArrayLayer :
                range.layerCount
        };
        // TODO: multisampling

        auto handle { this->descriptors_cpu_dsv.alloc() };

        D3D12_DEPTH_STENCIL_VIEW_DESC desc { format };
        if (type == VK_IMAGE_TYPE_2D && image->resource_desc.SampleDesc.Count != VK_SAMPLE_COUNT_1_BIT) {
            desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
            desc.Texture2DMS = D3D12_TEX2DMS_DSV { };
        } else if (type == VK_IMAGE_VIEW_TYPE_1D && range.baseArrayLayer == 0) {
            desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1D;
            desc.Texture1D = D3D12_TEX1D_DSV { range.baseMipLevel };
        } else if (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY) {
            desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
            desc.Texture1DArray = D3D12_TEX1D_ARRAY_DSV {
                range.baseMipLevel,
                range.baseArrayLayer,
                range.layerCount
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_2D && range.baseArrayLayer == 0) {
            desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            desc.Texture2D = D3D12_TEX2D_DSV { range.baseMipLevel };
        } else if (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_2D_ARRAY ||
                   type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
        {
            desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
            desc.Texture2DArray = D3D12_TEX2D_ARRAY_DSV {
                range.baseMipLevel,
                range.baseArrayLayer,
                range.layerCount,
            };
        }

        this->device->CreateDepthStencilView(resource, &desc, std::get<0>(handle));

        return handle;
    }

    auto create_shader_resource_view(
        image_t *image,
        VkImageViewType type,
        DXGI_FORMAT format,
        VkComponentMapping components,
        VkImageSubresourceRange const& range
    ) -> std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> {
        auto handle { this->descriptors_cpu_cbv_srv_uav.alloc() };
        auto resource { image->resource.Get() };

        const auto layer_count {
            range.layerCount == VK_REMAINING_ARRAY_LAYERS ?
                image->resource_desc.DepthOrArraySize - range.baseArrayLayer :
                range.layerCount
        };
        const auto level_count {
            range.levelCount == VK_REMAINING_MIP_LEVELS ?
                image->resource_desc.MipLevels - range.baseMipLevel :
                range.levelCount
        };

        // TODO: multisampling

        // TODO: other formats
        switch (format) {
            case DXGI_FORMAT_D16_UNORM: format = DXGI_FORMAT_R16_UNORM; break;
            case DXGI_FORMAT_D32_FLOAT: format = DXGI_FORMAT_R32_FLOAT; break;
            case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: {
                // Single bit only for SRV (descriptor set only view)
                if (range.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) {
                    format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
                } else {
                    format = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
                }
            } break;
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC desc { format };

        auto component = [] (VkComponentSwizzle component, D3D12_SHADER_COMPONENT_MAPPING identity) {
            switch (component) {
                case VK_COMPONENT_SWIZZLE_IDENTITY: return identity;
                case VK_COMPONENT_SWIZZLE_ZERO: return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0;
                case VK_COMPONENT_SWIZZLE_ONE: return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1;
                case VK_COMPONENT_SWIZZLE_R: return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0;
                case VK_COMPONENT_SWIZZLE_G: return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1;
                case VK_COMPONENT_SWIZZLE_B: return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2;
                case VK_COMPONENT_SWIZZLE_A: return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3;
                default: return identity;
            }
        };

        desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
            component(components.r, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0),
            component(components.g, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1),
            component(components.b, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2),
            component(components.a, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3)
        );

        if (type == VK_IMAGE_VIEW_TYPE_1D && range.baseArrayLayer == 0) {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            desc.Texture1D = D3D12_TEX1D_SRV {
                range.baseMipLevel,
                level_count,
                0.0 // TODO: ?
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY) {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            desc.Texture1DArray = D3D12_TEX1D_ARRAY_SRV {
                range.baseMipLevel,
                level_count,
                range.baseArrayLayer,
                layer_count,
                0.0 // TODO: ?
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_2D && range.baseArrayLayer == 0) {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            desc.Texture2D = D3D12_TEX2D_SRV {
                range.baseMipLevel,
                level_count,
                0,
                0.0 // TODO: ?
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_2D_ARRAY) {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            desc.Texture2DArray = D3D12_TEX2D_ARRAY_SRV {
                range.baseMipLevel,
                level_count,
                range.baseArrayLayer,
                layer_count,
                0,
                0.0 // TODO ?
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_CUBE && range.baseArrayLayer == 0) {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            desc.TextureCube = D3D12_TEXCUBE_SRV {
                range.baseMipLevel,
                level_count,
                0.0 // TODO ?
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
            desc.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV {
                range.baseMipLevel,
                level_count,
                range.baseArrayLayer,
                layer_count / 6,
                0.0 // TODO ?
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_3D) {
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            desc.Texture3D = D3D12_TEX3D_SRV {
                range.baseMipLevel,
                level_count,
                0.0 // TODO ?
            };
        }

        this->device->CreateShaderResourceView(resource, &desc, std::get<0>(handle));
        return handle;
    }

    auto create_unordered_access_view(
        ID3D12Resource* resource,
        VkImageViewType type,
        DXGI_FORMAT format,
        VkImageSubresourceRange const& range
    ) -> std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> {
        // TODO
        assert(range.layerCount != VK_REMAINING_ARRAY_LAYERS);

        auto handle { this->descriptors_cpu_cbv_srv_uav.alloc() };

        D3D12_UNORDERED_ACCESS_VIEW_DESC desc { format };
        if (type == VK_IMAGE_VIEW_TYPE_1D && range.baseArrayLayer == 0) {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
            desc.Texture1D = D3D12_TEX1D_UAV { range.baseMipLevel };
        } else if (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY) {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                range.baseMipLevel,
                range.baseArrayLayer,
                range.layerCount
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_2D && range.baseArrayLayer == 0) {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            desc.Texture2D = D3D12_TEX2D_UAV { range.baseMipLevel, 0 };
        } else if (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_2D_ARRAY ||
                   type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
        {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                range.baseMipLevel,
                range.baseArrayLayer,
                range.layerCount,
                0
            };
        } else if (type == VK_IMAGE_VIEW_TYPE_3D) {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Texture3D = D3D12_TEX3D_UAV { range.baseMipLevel, 0, ~0u };
        }

        this->device->CreateUnorderedAccessView(
            resource,
            nullptr, // counter
            &desc,
            std::get<0>(handle)
        );
        return handle;
    }

    auto destroy_render_target_view(std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> view) -> void {
        const auto [handle, index] = view;
        this->descriptors_cpu_rtv.free(handle, index);
    }

    auto destroy_depth_stencil_view(std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> view) -> void {
        const auto [handle, index] = view;
        this->descriptors_cpu_dsv.free(handle, index);
    }

    auto destroy_shader_resource_view(std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> view) -> void {
        const auto [handle, index] = view;
        this->descriptors_cpu_cbv_srv_uav.free(handle, index);
    }

    auto destroy_unordered_access_view(std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> view) -> void {
        const auto [handle, index] = view;
        this->descriptors_cpu_cbv_srv_uav.free(handle, index);
    }

private:
    /// Dispatchable object
    uintptr_t loader_magic;

public:
    ComPtr<ID3D12Device3> device;

    queue_t present_queue;
    std::vector<queue_t> general_queues;
    std::vector<queue_t> compute_queues;
    std::vector<queue_t> copy_queues;

    span<const heap_properties_t> heap_properties;

    // CPU descriptor heaps
    descriptors_cpu_t<D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 128> descriptors_cpu_rtv;
    descriptors_cpu_t<D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 128> descriptors_cpu_dsv;
    descriptors_cpu_t<D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 128> descriptors_cpu_cbv_srv_uav;
    descriptors_cpu_t<D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 128> descriptors_cpu_sampler;

    // GPU descriptor heaps
    descriptors_gpu_t<D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1'000'000> descriptors_gpu_cbv_srv_uav;
    descriptors_gpu_t<D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2048> descriptors_gpu_sampler;

    // Indirect execution signatures
    ComPtr<ID3D12CommandSignature> dispatch_indirect;
    std::unordered_map<uint32_t, ComPtr<ID3D12CommandSignature>> draw_indirect;
    std::unordered_map<uint32_t, ComPtr<ID3D12CommandSignature>> draw_indexed_indirect;
    // Indirect draw/draw_indirect signature access token
    std::shared_mutex draw_indirect_access;

    ComPtr<ID3D12PipelineState> pso_buffer_to_image { nullptr };
    ComPtr<ID3D12RootSignature> signature_buffer_to_image { nullptr };

    ComPtr<ID3D12PipelineState> pso_blit_2d { nullptr };
    ComPtr<ID3D12RootSignature> signature_blit_2d { nullptr };
};

class command_buffer_t {
public:
    struct pass_cache_t {
        // Current subpass
        size_t subpass;
        //
        render_pass_t* render_pass;
        framebuffer_t* framebuffer;
        //
        std::vector<VkClearValue> clear_values;
        D3D12_RECT render_area;
    };

    enum pipeline_slot_type {
        SLOT_GRAPHICS,
        SLOT_COMPUTE,
    };

    struct pipeline_slot_t {
    public:
        enum class data_entry_type {
            UNDEFINED, // TODO: Not needed probably
            SAMPLER,
            CBV_SRV_UAV,
            CONSTANT,
        };

        // Virtual root user data
        struct user_data_t {
        public:
            user_data_t() :
                dirty { 0 },
                data { 0 },
                type { data_entry_type::UNDEFINED }
            { }

            auto set_constant(size_t slot, uint32_t data) -> void {
                this->dirty.set(slot);
                this->data[slot] = data;
                this->type[slot] = data_entry_type::CONSTANT;
            }

            auto set_cbv_srv_uav(size_t slot, uint32_t data) -> void {
                this->dirty.set(slot);
                this->data[slot] = data;
                this->type[slot] = data_entry_type::CBV_SRV_UAV;
            }

            auto set_sampler(size_t slot, uint32_t data) -> void {
                this->dirty.set(slot);
                this->data[slot] = data;
                this->type[slot] = data_entry_type::SAMPLER;
            }

        public:
            std::bitset<D3D12_MAX_ROOT_COST> dirty;
            std::array<uint32_t, D3D12_MAX_ROOT_COST> data;
            std::array<data_entry_type, D3D12_MAX_ROOT_COST> type;
        };

    public:
        pipeline_slot_t() :
            pipeline { nullptr },
            signature { nullptr },
            root_data { }
        {}

    public:
        struct pipeline_t* pipeline;
        ID3D12RootSignature* signature;

        user_data_t root_data;
    };

public:
    command_buffer_t(ID3D12DescriptorHeap* heap_cbv_srv_uav, ID3D12DescriptorHeap* heap_sampler) :
        loader_magic { ICD_LOADER_MAGIC },
        command_list { nullptr },
        heap_cbv_srv_uav { heap_cbv_srv_uav },
        heap_sampler { heap_sampler },
        pass_cache { std::nullopt },
        active_slot { std::nullopt },
        active_pipeline { nullptr },
        dynamic_state_dirty { false },
        viewports_dirty { false },
        scissors_dirty { false },
        num_viewports_scissors { 0 },
        vertex_buffer_views_dirty { 0 },
        _device { nullptr },
        dynamic_state { },
        index_type { VK_INDEX_TYPE_UINT16 }
    { }

    ~command_buffer_t() {}

    auto operator->() -> ID3D12GraphicsCommandList2* {
        return this->command_list.Get();
    }

    auto reset() -> void {
        this->command_list->Reset(this->allocator, nullptr);

        this->pass_cache = std::nullopt;
        this->active_slot = std::nullopt;
        this->active_pipeline = nullptr;
        this->graphics_slot = pipeline_slot_t();
        this->compute_slot = pipeline_slot_t();
        this->num_viewports_scissors = 0;
        this->viewports_dirty = false;
        this->scissors_dirty = false;
        this->vertex_buffer_views_dirty = 0;
        this->dynamic_state = dynamic_state_t();
        this->dynamic_state_dirty = false;
        this->index_type = VK_INDEX_TYPE_UINT16;
    }

    auto end() -> void {
        this->command_list->Close();
        // TODO: error handling
    }

    auto device() -> ID3D12Device3* {
        return this->_device->device.Get();
    }

    auto update_user_data(
        pipeline_slot_t& slot,
        std::function<void (UINT slot, uint32_t data, UINT offset)> set_constant,
        std::function<void (UINT slot, D3D12_GPU_DESCRIPTOR_HANDLE handle)> set_table
    ) {
        auto user_data { slot.root_data };
        if (user_data.dirty.none()) {
            return;
        }

        const auto start_cbv_srv_uav { this->heap_cbv_srv_uav->GetGPUDescriptorHandleForHeapStart() };
        const auto start_sampler { this->heap_sampler->GetGPUDescriptorHandleForHeapStart() };

        const auto num_root_constant_entries { slot.pipeline->root_constants.size() };
        const auto num_table_entries { slot.pipeline->num_signature_entries };

        auto start_constant { 0u };
        for (auto i : range(num_root_constant_entries)) {
            auto const& root_constant { slot.pipeline->root_constants[i] };

            for (auto c : range(start_constant, start_constant + root_constant.size / 4)) {
                if (user_data.dirty[c]) {
                    set_constant(
                        static_cast<UINT>(i),
                        user_data.data[c],
                        c - start_constant
                    );
                    user_data.dirty.reset(c);
                }
            }

            start_constant += root_constant.size / 4;
        }

        for (auto i : range(num_table_entries)) {
            const auto data_slot { slot.pipeline->num_root_constants + i };
            if (user_data.dirty[data_slot]) {
                const auto offset { user_data.data[data_slot] };
                SIZE_T descriptor_start { 0u };
                switch (user_data.type[data_slot]) {
                    case pipeline_slot_t::data_entry_type::CBV_SRV_UAV: descriptor_start = start_cbv_srv_uav.ptr; break;
                    case pipeline_slot_t::data_entry_type::SAMPLER: descriptor_start = start_sampler.ptr; break;
                    default: WARN("Unexpected user data entry: {}", static_cast<uint32_t>(user_data.type[i]));
                }

                const auto handle { D3D12_GPU_DESCRIPTOR_HANDLE { descriptor_start + offset } };
                set_table(
                    static_cast<UINT>(num_root_constant_entries + i),
                    handle
                );
                user_data.dirty.reset(data_slot);
            }
        }
    }

    auto bind_graphics_slot(draw_type draw_type) -> void {
        if (!this->active_slot) {
            std::array<ID3D12DescriptorHeap *const, 2> heaps {
                this->heap_cbv_srv_uav,
                this->heap_sampler
            };
            this->command_list->SetDescriptorHeaps(2, &heaps[0]);
        }


        if (
            (this->dynamic_state_dirty && this->graphics_slot.pipeline->dynamic_states) ||
            this->active_slot != SLOT_GRAPHICS // Check if we are switching to Graphics
        ) {
            this->active_pipeline = nullptr;
        }

        if (!this->active_pipeline) {
            this->command_list->SetPipelineState(
                std::visit(
                    stdx::match(
                        [] (pipeline_t::unique_pso_t& pso) {
                            return pso.pipeline.Get();
                        },
                        [&] (pipeline_t::dynamic_pso_t& pso) {
                            // Check if we have one available already
                            {
                                auto dynamic_state { this->dynamic_state };
                                this->graphics_slot.pipeline->pso_access.lock_shared();
                                switch (draw_type) {
                                    case draw_type::DRAW: {
                                        // Check all three strip cut variants
                                        dynamic_state.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
                                        auto pipeline { pso.pipelines.find(dynamic_state) };
                                        if (pipeline != pso.pipelines.end()) {
                                            return pipeline->second.Get();
                                        }
                                        dynamic_state.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
                                        pipeline = pso.pipelines.find(dynamic_state);
                                        if (pipeline != pso.pipelines.end()) {
                                            return pipeline->second.Get();
                                        }
                                        dynamic_state.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
                                        pipeline = pso.pipelines.find(dynamic_state);
                                        if (pipeline != pso.pipelines.end()) {
                                            return pipeline->second.Get();
                                        }
                                    } break;
                                    case draw_type::DRAW_INDEXED: {
                                        if (graphics_slot.pipeline->dynamic_states & DYNAMIC_STATE_PRIMITIVE_RESTART) {
                                            switch (index_type) {
                                                case VK_INDEX_TYPE_UINT16:
                                                    dynamic_state.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
                                                    break;
                                                case VK_INDEX_TYPE_UINT32:
                                                    dynamic_state.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
                                                    break;
                                            }
                                        } else {
                                            dynamic_state.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
                                        }

                                        auto pipeline { pso.pipelines.find(dynamic_state) };
                                        if (pipeline != pso.pipelines.end()) {
                                            return pipeline->second.Get();
                                        }
                                    } break;
                                }
                                this->graphics_slot.pipeline->pso_access.unlock_shared();
                            }

                            // Generate a new one
                            auto shader_bc = [] (ComPtr<ID3DBlob> shader) {
                                return shader ?
                                    D3D12_SHADER_BYTECODE { shader->GetBufferPointer(), shader->GetBufferSize() } :
                                    D3D12_SHADER_BYTECODE { 0, 0 };
                            };

                            // Apply currently set dynamic state values
                            const auto dynamics { this->graphics_slot.pipeline->dynamic_states };
                            auto const& dynamic_state { this->dynamic_state };

                            auto rasterizer_state { pso.rasterizer_state };
                            if (dynamics & DYNAMIC_STATE_DEPTH_BIAS) {
                                rasterizer_state.DepthBias = dynamic_state.depth_bias;
                                rasterizer_state.DepthBiasClamp = dynamic_state.depth_bias_clamp;
                                rasterizer_state.SlopeScaledDepthBias = dynamic_state.depth_bias_slope;
                            }

                            auto depth_stencil_state { pso.depth_stencil_state };
                            if (dynamics & DYNAMIC_STATE_STENCIL_COMPARE_MASK) {
                                depth_stencil_state.StencilReadMask = dynamic_state.stencil_read_mask;
                            }
                            if (dynamics & DYNAMIC_STATE_STENCIL_WRITE_MASK) {
                                depth_stencil_state.StencilWriteMask = dynamic_state.stencil_write_mask;
                            }

                            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc {
                                this->graphics_slot.pipeline->signature,
                                shader_bc(pso.vertex_shader),
                                shader_bc(pso.pixel_shader),
                                shader_bc(pso.domain_shader),
                                shader_bc(pso.hull_shader),
                                shader_bc(pso.geometry_shader),
                                D3D12_STREAM_OUTPUT_DESC { }, // not used
                                pso.blend_state,
                                pso.sample_mask,
                                rasterizer_state,
                                depth_stencil_state,
                                D3D12_INPUT_LAYOUT_DESC {
                                    pso.input_elements.data(),
                                    static_cast<UINT>(pso.input_elements.size())
                                },
                                dynamic_state.strip_cut,
                                pso.topology_type,
                                pso.num_render_targets,
                                pso.rtv_formats[0], pso.rtv_formats[1], pso.rtv_formats[2], pso.rtv_formats[3],
                                pso.rtv_formats[4], pso.rtv_formats[5], pso.rtv_formats[6], pso.rtv_formats[7],
                                pso.dsv_format,
                                pso.sample_desc,
                                0, // NodeMask
                                D3D12_CACHED_PIPELINE_STATE { }, // TODO
                                D3D12_PIPELINE_STATE_FLAG_NONE, // TODO
                            };

                            ComPtr<ID3D12PipelineState> pipeline;
                            auto const hr { this->device()->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline)) };
                            // TODO: error handling

                            this->graphics_slot.pipeline->pso_access.lock();
                            pso.pipelines.emplace(dynamic_state, pipeline);
                            this->graphics_slot.pipeline->pso_access.unlock();

                            return pipeline.Get();
                        }
                    ),
                    this->graphics_slot.pipeline->pso
                )
            );
        }

        this->active_slot = SLOT_GRAPHICS;

        if (this->viewports_dirty) {
            this->command_list->RSSetViewports(
                this->num_viewports_scissors,
                this->viewports
            );
            this->viewports_dirty = false;
        }

        if (this->scissors_dirty) {
            this->command_list->RSSetScissorRects(
                this->num_viewports_scissors,
                this->scissors
            );
            this->scissors_dirty = false;
        }

        if (this->vertex_buffer_views_dirty.any()) {
            std::optional<size_t> in_range { std::nullopt };
            for (auto i : range(MAX_VERTEX_BUFFER_SLOTS)) {
                const auto has_flag { this->vertex_buffer_views_dirty[i] };

                if (has_flag) {
                    this->vertex_buffer_views_dirty.reset(i);
                    if (!in_range) {
                        in_range = i;
                    }
                } else if (in_range) {
                    this->command_list->IASetVertexBuffers(
                        static_cast<UINT>(*in_range),
                        static_cast<UINT>(i - *in_range),
                        this->vertex_buffer_views
                    );
                    in_range = std::nullopt;
                }
            }

            if (in_range) {
                this->command_list->IASetVertexBuffers(
                    static_cast<UINT>(*in_range),
                    static_cast<UINT>(MAX_VERTEX_BUFFER_SLOTS - *in_range),
                    this->vertex_buffer_views
                );
            }
        }

        update_user_data(
            this->graphics_slot,
            [&] (UINT slot, uint32_t data, UINT offset) {
                this->command_list->SetGraphicsRoot32BitConstant(
                    slot,
                    data,
                    offset
                );
            },
            [&] (UINT slot, D3D12_GPU_DESCRIPTOR_HANDLE handle) {
                this->command_list->SetGraphicsRootDescriptorTable(
                    slot,
                    handle
                );
            }
        );
    }

    auto bind_compute_slot() -> void {
        if (!this->active_slot) {
            std::array<ID3D12DescriptorHeap *const, 2> heaps {
                this->heap_cbv_srv_uav,
                this->heap_sampler
            };
            this->command_list->SetDescriptorHeaps(2, &heaps[0]);
        }

        if (this->active_slot != SLOT_COMPUTE) {
            this->active_pipeline = nullptr;
            this->active_slot = SLOT_COMPUTE;
        }

        if (!this->active_pipeline) {
            this->command_list->SetPipelineState(
                std::get<pipeline_t::unique_pso_t>(this->compute_slot.pipeline->pso).pipeline.Get()
            );
        }

        update_user_data(
            this->compute_slot,
            [&] (UINT slot, uint32_t data, UINT offset) {
                this->command_list->SetComputeRoot32BitConstant(
                    slot,
                    data,
                    offset
                );
            },
            [&] (UINT slot, D3D12_GPU_DESCRIPTOR_HANDLE handle) {
                this->command_list->SetComputeRootDescriptorTable(
                    slot,
                    handle
                );
            }
        );
    }


    auto begin_subpass(VkSubpassContents contents) -> void {
        // TODO: contents

        auto const& pass_cache { *this->pass_cache };
        auto clear_values { span<const VkClearValue>(pass_cache.clear_values) };
        auto framebuffer { pass_cache.framebuffer };
        auto render_pass { pass_cache.render_pass };

        const auto subpass_id { pass_cache.subpass };
        auto const& subpass { render_pass->subpasses[subpass_id] };

        // Clear attachments on first use
        for (auto const& color_attachment : subpass.color_attachments) {
            if (color_attachment.attachment == VK_ATTACHMENT_UNUSED) {
                continue;
            }

            auto const& attachment { render_pass->attachments[color_attachment.attachment] };
            if (attachment.first_use == subpass_id) {
                auto view { framebuffer->attachments[color_attachment.attachment] };
                if (attachment.desc.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
                    // TODO: temp barriers...
                    this->command_list->ResourceBarrier(1,
                        &CD3DX12_RESOURCE_BARRIER::Transition(view->image, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)
                    );
                    this->command_list->ClearRenderTargetView(
                        std::get<0>(*view->rtv),
                        clear_values[color_attachment.attachment].color.float32,
                        1,
                        &pass_cache.render_area
                    );
                }
            }
        }

        const auto depth_attachment { subpass.depth_attachment.attachment };
        if (depth_attachment != VK_ATTACHMENT_UNUSED) {
            auto const& attachment {
                render_pass->attachments[depth_attachment]
            };

            if (attachment.first_use == subpass_id) {
                auto view { framebuffer->attachments[depth_attachment] };

                D3D12_CLEAR_FLAGS clear_flags { static_cast<D3D12_CLEAR_FLAGS>(0) };
                float depth { 0 };
                UINT8 stencil { 0 };

                if (attachment.desc.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
                    clear_flags |= D3D12_CLEAR_FLAG_DEPTH;
                    depth = clear_values[depth_attachment].depthStencil.depth;
                }

                // TODO: stencil

                if (clear_flags) {
                    // TODO: temp barriers...
                    this->command_list->ResourceBarrier(1,
                        &CD3DX12_RESOURCE_BARRIER::Transition(view->image, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE)
                    );
                    this->command_list->ClearDepthStencilView(
                        std::get<0>(*view->dsv),
                        clear_flags,
                        depth,
                        stencil,
                        1,
                        &pass_cache.render_area
                    );
                }
            }
        }

        // Bind render targets
        auto color_attachments { span<const VkAttachmentReference>(subpass.color_attachments) };
        const auto num_rtvs { color_attachments.size() };
        D3D12_CPU_DESCRIPTOR_HANDLE render_targets[D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT];
        for (auto i : range(num_rtvs)) {
            // TODO: support VK_ATTACHMENT_UNUSED
            const auto color_attachment { color_attachments[i].attachment };
            assert(color_attachment != VK_ATTACHMENT_UNUSED);
            render_targets[i] = std::get<0>(*framebuffer->attachments[color_attachment]->rtv);
        }

        D3D12_CPU_DESCRIPTOR_HANDLE* dsv { nullptr };
        if (depth_attachment != VK_ATTACHMENT_UNUSED) {
            dsv = &std::get<0>(*framebuffer->attachments[depth_attachment]->dsv);
        }

        this->command_list->OMSetRenderTargets(
            static_cast<UINT>(num_rtvs),
            render_targets,
            !!dsv,
            dsv
        );
    }

    auto end_subpass() -> void {
        auto const& pass_cache { *this->pass_cache };
        auto framebuffer { pass_cache.framebuffer };
        auto render_pass { pass_cache.render_pass };

        const auto subpass_id { pass_cache.subpass };
        auto const& subpass { render_pass->subpasses[subpass_id] };

        // Clear attachments on first use
        // TODO
        for (auto const& color_attachment : subpass.color_attachments) {
            if (color_attachment.attachment == VK_ATTACHMENT_UNUSED) {
                continue;
            }

            auto const& attachment { render_pass->attachments[color_attachment.attachment] };
            if (attachment.first_use == subpass_id) {
                auto view { framebuffer->attachments[color_attachment.attachment] };
                if (attachment.desc.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
                    // TODO: temp barriers...
                    this->command_list->ResourceBarrier(1,
                        &CD3DX12_RESOURCE_BARRIER::Transition(view->image, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON)
                    );
                }
            }
        }

        const auto depth_attachment { subpass.depth_attachment.attachment };
        if (depth_attachment != VK_ATTACHMENT_UNUSED) {
            auto const& attachment {
                render_pass->attachments[depth_attachment]
            };

            if (attachment.first_use == subpass_id) {
                auto view { framebuffer->attachments[depth_attachment] };

                if (attachment.desc.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
                    // TODO: temp barriers...
                    this->command_list->ResourceBarrier(1,
                        &CD3DX12_RESOURCE_BARRIER::Transition(view->image, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COMMON)
                    );
                }
            }
        }

        for (auto i : range(subpass.resolve_attachments.size())) {
            auto const& color_attachment { subpass.color_attachments[i] };
            auto const& resolve_attachment { subpass.resolve_attachments[i] };

            if (resolve_attachment.attachment == VK_ATTACHMENT_UNUSED) {
                continue;
            }

            auto color_view { framebuffer->attachments[color_attachment.attachment] };
            auto resolve_view { framebuffer->attachments[resolve_attachment.attachment] };

            this->command_list->ResolveSubresource(
                resolve_view->image,
                0, // TODO: D3D12CalcSubresource(resolve_view->)
                color_view->image,
                0, // TODO
                formats[render_pass->attachments[color_attachment.attachment].desc.format]
            );
        }
    }

private:
    /// Dispatchable object
    uintptr_t loader_magic;

public:
    //
    ComPtr<ID3D12GraphicsCommandList2> command_list;
    std::optional<pipeline_slot_type> active_slot;
    ID3D12PipelineState* active_pipeline;

    ID3D12DescriptorHeap* heap_cbv_srv_uav;
    ID3D12DescriptorHeap* heap_sampler;

    bool dynamic_state_dirty;
    bool viewports_dirty;
    bool scissors_dirty;
    std::bitset<MAX_VERTEX_BUFFER_SLOTS> vertex_buffer_views_dirty;

    // Currently set dynamic state
    dynamic_state_t dynamic_state;
    VkIndexType index_type;

    device_t* _device;

    // Owning command allocator, required for reset
    ID3D12CommandAllocator* allocator;

    std::optional<pass_cache_t> pass_cache;

    pipeline_slot_t graphics_slot;
    pipeline_slot_t compute_slot;

    D3D12_VIEWPORT viewports[D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
    D3D12_RECT scissors[D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
    UINT num_viewports_scissors;

    // Stride values are not known when calling `vkCmdBindVertexBuffers`
    D3D12_VERTEX_BUFFER_VIEW vertex_buffer_views[MAX_VERTEX_BUFFER_SLOTS];

    std::vector<ComPtr<ID3D12DescriptorHeap>> temp_heaps;
};

struct semaphore_t {
    ComPtr<ID3D12Fence> fence;
};

struct surface_t {
    ComPtr<IDXGIFactory5> dxgi_factory;
    HWND hwnd;
};

struct swapchain_t {
    ComPtr<IDXGISwapChain3> swapchain;
    std::vector<image_t> images;
};

struct buffer_t {
    ComPtr<ID3D12Resource> resource;
    VkMemoryRequirements memory_requirements;
    D3D12_RESOURCE_FLAGS usage_flags;
};

struct device_memory_t {
    ComPtr<ID3D12Heap> heap;
    ComPtr<ID3D12Resource> buffer;
    VkDeviceSize size;
};

struct pipeline_cache_t {

};

struct fence_t {
    ComPtr<ID3D12Fence> fence;
};

struct descriptor_set_layout_t {
    struct binding_t {
        uint32_t binding;
        VkDescriptorType type;
        uint32_t descriptor_count;
        VkShaderStageFlags stage_flags;
        std::vector<VkSampler> immutable_samplers;
    };

    std::vector<binding_t> layouts;
};

struct descriptor_set_t {
    using binding_index = size_t;

    struct binding_t {
        D3D12_CPU_DESCRIPTOR_HANDLE start_cbv_srv_uav;
        D3D12_CPU_DESCRIPTOR_HANDLE start_sampler;
        size_t num_descriptors;
    };

    // A descriptor sets is a subslice fo a descriptor pool.
    // Needed when binding descriptor sets.
    std::optional<D3D12_GPU_DESCRIPTOR_HANDLE> start_cbv_srv_uav;
    std::optional<D3D12_GPU_DESCRIPTOR_HANDLE> start_sampler;

    // Each binding of the descriptor set is again as slice of the descriptor set slice.
    // Needed for updating descriptor sets.
    std::map<binding_index, binding_t> bindings;
};

struct descriptor_pool_t {
public:
    descriptor_pool_t(size_t num_cbv_srv_uav, size_t num_sampler) :
        slice_cbv_srv_uav { num_cbv_srv_uav },
        slice_sampler { num_sampler }
    { }

    auto alloc(size_t num_cbv_srv_uav, size_t num_sampler)
        -> std::tuple<descriptor_cpu_gpu_handle_t, descriptor_cpu_gpu_handle_t>
    {
        const auto start_cbv_srv_uav = this->slice_cbv_srv_uav.alloc(num_cbv_srv_uav);
        const auto start_sampler = this->slice_sampler.alloc(num_sampler);

        return std::make_tuple(start_cbv_srv_uav, start_sampler);
    }

public:
    struct slice_t {
    public:
        slice_t(size_t num) :
            allocator { num }
        {}

        auto alloc(size_t num) -> descriptor_cpu_gpu_handle_t {
            const auto range = this->allocator.alloc(num);
            if (range) {
                const auto [start_cpu, start_gpu] = this->start;
                return std::make_tuple(
                    D3D12_CPU_DESCRIPTOR_HANDLE { start_cpu.ptr + this->handle_size * range->_start },
                    D3D12_GPU_DESCRIPTOR_HANDLE { start_gpu.ptr + this->handle_size * range->_start }
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
    public:
        descriptor_cpu_gpu_handle_t start;
        UINT handle_size;
        free_list allocator;
    };

    slice_t slice_cbv_srv_uav;
    slice_t slice_sampler;
};

enum root_table_flags {
    TABLE_CBV_SRV_UAV = 0x1,
    TABLE_SAMPLER = 0x2,
};

struct pipeline_layout_t {
    ComPtr<ID3D12RootSignature> signature;
    std::vector<uint32_t> tables; // root_table_flags
    std::vector<VkPushConstantRange> root_constants;
    size_t num_root_constants; // Number of root constants (32bit)
    size_t num_signature_entries;
};

struct sampler_t {
    std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t> sampler;
};

auto translate_spirv (
    VkPipelineShaderStageCreateInfo const& stage,
    pipeline_layout_t* layout
) {
    std::string entry_name;
    std::string shader_code;

    auto module { reinterpret_cast<shader_module_t *>(stage.module) };

    try {
        // Translate SPIR-V to HLSL code
        spirv_cross::CompilerHLSL compiler { module->spirv.data(), module->spirv.size() };
        compiler.set_options(spirv_cross::CompilerHLSL::Options {
            51, // shader model 5.1
            true, // point size builtin
            true, // point coord builtin
        });
        spirv_cross::CompilerGLSL::Options glsl_options;
        glsl_options.vertex.flip_vert_y = true;
        compiler.spirv_cross::CompilerGLSL::set_options(glsl_options);

        // Path shader resource interface
        auto shader_resources { compiler.get_shader_resources() };
        for (auto const& image : shader_resources.separate_images) {
            auto set { compiler.get_decoration(image.id, spv::Decoration::DecorationDescriptorSet) };
            compiler.set_decoration(image.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set);
        }
        for (auto const& uniform_buffer : shader_resources.uniform_buffers) {
            auto set { compiler.get_decoration(uniform_buffer.id, spv::Decoration::DecorationDescriptorSet) };
            compiler.set_decoration(uniform_buffer.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set);
        }
        for (auto const& storage_buffer : shader_resources.storage_buffers) {
            auto set { compiler.get_decoration(storage_buffer.id, spv::Decoration::DecorationDescriptorSet) };
            compiler.set_decoration(storage_buffer.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set);
        }
        for (auto const& storage_image : shader_resources.storage_images) {
            auto set { compiler.get_decoration(storage_image.id, spv::Decoration::DecorationDescriptorSet) };
            compiler.set_decoration(storage_image.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set);
        }
        for (auto const& image : shader_resources.sampled_images) {
            auto set { compiler.get_decoration(image.id, spv::Decoration::DecorationDescriptorSet) };
            compiler.set_decoration(image.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set); // Sampler offset done in spirv-cross
        }
        for (auto const& sampler : shader_resources.separate_samplers) {
            auto set { compiler.get_decoration(sampler.id, spv::Decoration::DecorationDescriptorSet) };
            compiler.set_decoration(sampler.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set + 1);
        }

        // TODO: more shader resource supports!

        // Push constants
        std::vector<spirv_cross::RootConstants> root_constants {};
        for (auto const& root_constant : layout->root_constants) {
            if (root_constant.stageFlags & stage.stage) {
                root_constants.emplace_back(
                    spirv_cross::RootConstants {
                        root_constant.offset,
                        root_constant.offset + root_constant.size,
                        root_constant.offset / 4,
                        PUSH_CONSTANT_REGISTER_SPACE
                    }
                );
            }
        }
        compiler.set_root_constant_layouts(root_constants);

        // Specialization constants
        if (stage.pSpecializationInfo) {
            auto const& specialization_info { *stage.pSpecializationInfo };
            auto map_entries {
                span<const VkSpecializationMapEntry>(specialization_info.pMapEntries, specialization_info.mapEntryCount)
            };
            auto data {
                span<const uint8_t>(
                    static_cast<const uint8_t *>(specialization_info.pData),
                    specialization_info.dataSize
                )
            };

            auto spec_constants { compiler.get_specialization_constants() };

            for (auto const& map_entry : map_entries) {
                auto spec_constant {
                    std::find_if(
                        spec_constants.begin(),
                        spec_constants.end(),
                        [&] (spirv_cross::SpecializationConstant const& sc) {
                            return sc.constant_id == map_entry.constantID;
                        }
                    )
                };

                if (spec_constant != spec_constants.end())  {
                    auto& constant { compiler.get_constant(spec_constant->id) };
                    switch (map_entry.size) {
                        case 4: {
                            constant.m.c[0].r[0].u32 =
                                *(reinterpret_cast<const uint32_t *>(&data[map_entry.offset]));
                        } break;
                        case 8: {
                            constant.m.c[0].r[0].u64 =
                                *(reinterpret_cast<const uint64_t *>(&data[map_entry.offset]));
                        } break;
                        default: WARN("Unexpected specialization constant size: {}", map_entry.size);
                    }
                }
            }
        }

        shader_code = compiler.compile();
        entry_name = compiler.get_cleansed_entry_point_name(stage.pName);

        DEBUG("SPIRV-Cross generated shader: {}", shader_code);
    } catch(spirv_cross::CompilerError err) {
        ERR("SPIRV-Cross translation error: {}", err.what());
    }

    return std::make_tuple(entry_name, shader_code);
};

auto compile_shader(std::string_view stage, std::string_view entry, std::string_view shader) {
    ComPtr<ID3DBlob> shader_blob { nullptr };
    ComPtr<ID3DBlob> error { nullptr };

    const auto hr {
        D3DCompile(
            shader.data(),
            shader.size(),
            nullptr, // TODO
            nullptr, // TODO
            nullptr, // TODO
            entry.data(),
            stage.data(),
            1,
            D3DCOMPILE_DEBUG, // TODO: pipeline flags
            &shader_blob,
            &error
        )
    };

    if (FAILED(hr)) {
        ERR("D3DCompile error: {} {}",
            hr, static_cast<const char *>(error->GetBufferPointer()));
    }

    return shader_blob;
};

auto create_command_signature(
    ID3D12Device* device,
    D3D12_INDIRECT_ARGUMENT_TYPE type,
    UINT stride
) -> ComPtr<ID3D12CommandSignature> {
    const D3D12_INDIRECT_ARGUMENT_DESC argument { type };
    const D3D12_COMMAND_SIGNATURE_DESC desc {
        stride,
        1,
        &argument,
        0u,
    };

    ComPtr<ID3D12CommandSignature> signature { nullptr };
    auto const hr {
        device->CreateCommandSignature(
            &desc,
            nullptr, // not required
            IID_PPV_ARGS(&signature)
        )
    };
    // TODO: error handling

    return signature;
}

// ------------------------------------- API implementations
VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkInstance *pInstance
) {
    *pInstance = reinterpret_cast<VkInstance>(new instance_t());
    TRACE("vkCreateInstance");
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(
    VkInstance                                  instance,
    const VkAllocationCallbacks*                pAllocator
) {
    // TODO: delete reinterpret_cast<instance_t *>(instance);
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(
    VkInstance                                  _instance,
    uint32_t*                                   pPhysicalDeviceCount,
    VkPhysicalDevice*                           pPhysicalDevices
) {
    TRACE("vkEnumeratePhysicalDevices");

    auto instance { reinterpret_cast<instance_t *>(_instance) };
    if (!pPhysicalDevices) {
        *pPhysicalDeviceCount = static_cast<uint32_t>(instance->adapters().size());
        return VK_SUCCESS;
    }

    const auto adapters { instance->adapters() };
    const auto num_adapters { adapters.size() };
    for (auto i : range(num_adapters)) {
        pPhysicalDevices[i] = (VkPhysicalDevice)(&adapters[i]);
    }

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceFeatures*                   pFeatures
) {
    TRACE("vkGetPhysicalDeviceFeatures");

    // TODO
    *pFeatures = VkPhysicalDeviceFeatures {
        true, // robustBufferAccess
        true, // fullDrawIndexUint32
        true, // imageCubeArray
        true, // independentBlend
        false, // geometryShader // TODO
        false, // tessellationShader // TODO
        false, // sampleRateShading // TODO ?
        true, // dualSrcBlend
        false, // logicOp // TODO: optional for 11_0
        true, // multiDrawIndirect
        true, // drawIndirectFirstInstance
        true, // depthClamp
        true, // depthBiasClamp
        true, // fillModeNonSolid // TODO: point is still hairy
        false, // depthBounds // TODO: check for support
        false, // wideLines
        false, // largePoints // TODO
        true, // alphaToOne
        true, // multiViewport
        true, // samplerAnisotropy
        false, // textureCompressionETC2
        false, // textureCompressionASTC_LDR
        true, // textureCompressionBC

        // TODO
        false, // occlusionQueryPrecise
        false, // pipelineStatisticsQuery
        false, // vertexPipelineStoresAndAtomics
        false, // fragmentStoresAndAtomics
        false, // shaderTessellationAndGeometryPointSize
        false, // shaderImageGatherExtended
        false, // shaderStorageImageExtendedFormats
        false, // shaderStorageImageMultisample
        false, // shaderStorageImageReadWithoutFormat
        false, // shaderStorageImageWriteWithoutFormat
        false, // shaderUniformBufferArrayDynamicIndexing
        false, // shaderSampledImageArrayDynamicIndexing
        false, // shaderStorageBufferArrayDynamicIndexing
        false, // shaderStorageImageArrayDynamicIndexing
        false, // shaderClipDistance
        false, // shaderCullDistance
        false, // shaderFloat64
        false, // shaderInt64
        false, // shaderInt16
        false, // shaderResourceResidency
        false, // shaderResourceMinLod
        false, // sparseBinding
        false, // sparseResidencyBuffer
        false, // sparseResidencyImage2D
        false, // sparseResidencyImage3D
        false, // sparseResidency2Samples
        false, // sparseResidency4Samples
        false, // sparseResidency8Samples
        false, // sparseResidency16Samples
        false, // sparseResidencyAliased
        false, // variableMultisampleRate
        false, // inheritedQueries
    };
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(
    VkPhysicalDevice                            physicalDevice,
    VkFormat                                    format,
    VkFormatProperties*                         pFormatProperties
) {
    TRACE("vkGetPhysicalDeviceFormatProperties");

    // TODO
    *pFormatProperties = formats_property[format];
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice                            physicalDevice,
    VkFormat                                    format,
    VkImageType                                 type,
    VkImageTiling                               tiling,
    VkImageUsageFlags                           usage,
    VkImageCreateFlags                          flags,
    VkImageFormatProperties*                    pImageFormatProperties
) {
    WARN("vkGetPhysicalDeviceImageFormatProperties unimplemented");

    // TODO
    *pImageFormatProperties = VkImageFormatProperties {
        VkExtent3D {
            D3D12_REQ_TEXTURE1D_U_DIMENSION,
            D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
            D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
        },
        1,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        1,
    };

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(
    VkPhysicalDevice                            _physicalDevice,
    VkPhysicalDeviceProperties*                 pProperties
) {
    TRACE("vkGetPhysicalDeviceProperties");

    auto physical_device { reinterpret_cast<physical_device_t *>(_physicalDevice) };

    *pProperties = VkPhysicalDeviceProperties {
        physical_device->properties.api_version,
        physical_device->properties.driver_version,
        physical_device->properties.vendor_id,
        physical_device->properties.device_id,
        physical_device->properties.device_type,
        "",
        { }, // TODO pipeline cache UUID
        physical_device->limits,
        { }, // TODO: sparse properties
    };
    std::memcpy(pProperties->deviceName, physical_device->properties.device_name.data(), VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pQueueFamilyPropertyCount,
    VkQueueFamilyProperties*                    pQueueFamilyProperties
) {
    TRACE("vkGetPhysicalDeviceQueueFamilyProperties");

    // TODO
    std::array<VkQueueFamilyProperties, 1> queue_family_properties {{
        {
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, 1
        }

        /*
        {
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, 1
        },
        {
            VK_QUEUE_COMPUTE_BIT, 1
        },
        {
            VK_QUEUE_TRANSFER_BIT, 1
        }
        */
    }};

    if (!pQueueFamilyProperties) {
        *pQueueFamilyPropertyCount = static_cast<uint32_t>(queue_family_properties.size());
    } else {
        auto num_properties {
            std::min<uint32_t>(
                *pQueueFamilyPropertyCount,
                static_cast<uint32_t>(queue_family_properties.size())
            )
        };
        for (auto i : range(num_properties)) {
            pQueueFamilyProperties[i] = queue_family_properties[i];
        }
    }

    log()->warn("vkGetPhysicalDeviceQueueFamilyProperties unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice                            _physicalDevice,
    VkPhysicalDeviceMemoryProperties*           pMemoryProperties
) {
    TRACE("vkGetPhysicalDeviceMemoryProperties");

    auto physical_device { reinterpret_cast<physical_device_t *>(_physicalDevice) };
    *pMemoryProperties = physical_device->memory_properties;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(
    VkInstance                                  instance,
    const char*                                 pName
) {
    VK_INSTANCE_FNC()
    VK_PHYSICAL_DEVICE_FNC()
    VK_DEVICE_FNC()

    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(
    VkDevice                                    device,
    const char*                                 pName
) {
    TRACE("vkGetDeviceProcAddr");

    VK_DEVICE_FNC()

    DEBUG("{}", pName);

    return nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(
    VkPhysicalDevice                            _physicalDevice,
    const VkDeviceCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDevice*                                   pDevice
) {
    TRACE("vkCreateDevice");

    // TODO: handle lock on device destruction
    auto physical_device { reinterpret_cast<physical_device_t *>(_physicalDevice) };
    if (!physical_device->is_open.try_lock()) {
        return VK_ERROR_TOO_MANY_OBJECTS;
    }

    auto const& info { *pCreateInfo };
    span<const VkDeviceQueueCreateInfo> queue_infos { info.pQueueCreateInfos, info.queueCreateInfoCount };

    // Create device
    ComPtr<ID3D12Device3> raw_device;

    {
        const auto hr { ::D3D12CreateDevice(
            physical_device->adapter.Get(),
            D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&raw_device)
        )};

        if (!SUCCEEDED(hr)) {
            ERR("error on device creation: {}", hr);
        }
    }

    auto device = new device_t(raw_device);
    device->heap_properties = physical_device->heap_properties;

    auto create_queue = [&] (const uint32_t index, const float priority) {
        D3D12_COMMAND_LIST_TYPE type;
        switch (index) {
            case QUEUE_FAMILY_GENERAL_PRESENT:
            case QUEUE_FAMILY_GENERAL: type = D3D12_COMMAND_LIST_TYPE_DIRECT; break;
            case QUEUE_FAMILY_COMPUTE: type = D3D12_COMMAND_LIST_TYPE_COMPUTE; break;
            case QUEUE_FAMILY_COPY: type = D3D12_COMMAND_LIST_TYPE_COPY; break;
            default: ERR("Unexpected queue family index: {}", index);
        }
        const D3D12_COMMAND_QUEUE_DESC desc {
            type,
            priority < 0.5 ?
                D3D12_COMMAND_QUEUE_PRIORITY_NORMAL :
                D3D12_COMMAND_QUEUE_PRIORITY_HIGH, // TODO: continuous?
            D3D12_COMMAND_QUEUE_FLAG_NONE,
            0
        };

        ComPtr<ID3D12CommandQueue> queue { nullptr };
        {
            auto const hr { (*device)->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)) };
            // TODO: error handling
        }

        return ComPtr<ID3D12CommandQueue>(queue);
    };

    for (auto const& queue_info : queue_infos) {
        const auto family_index { queue_info.queueFamilyIndex };

        ComPtr<ID3D12Fence> idle_fence { nullptr };
        {
            const auto hr {
                (*device)->CreateFence(
                    0,
                    D3D12_FENCE_FLAG_NONE,
                    IID_PPV_ARGS(&idle_fence)
                )
            };
            // TODO: error handling
        }

        switch (family_index) {
            case QUEUE_FAMILY_GENERAL_PRESENT: {
                device->present_queue = { create_queue(family_index, queue_info.pQueuePriorities[0]), idle_fence};
            } break;
            case QUEUE_FAMILY_GENERAL: {
                for (auto i : range(queue_info.queueCount)) {
                    device->general_queues.emplace_back(
                        create_queue(family_index, queue_info.pQueuePriorities[i]),
                        idle_fence
                    );
                }
            } break;
            case QUEUE_FAMILY_COMPUTE: {
                for (auto i : range(queue_info.queueCount)) {
                    device->compute_queues.emplace_back(
                        create_queue(family_index, queue_info.pQueuePriorities[i]),
                        idle_fence
                    );
                }
            } break;
            case QUEUE_FAMILY_COPY: {
                for (auto i : range(queue_info.queueCount)) {
                    device->copy_queues.emplace_back(
                        create_queue(family_index, queue_info.pQueuePriorities[i]),
                        idle_fence
                    );
                }
            } break;
            default:
                ERR("Unexpected queue family index: {}", family_index);
        }
    }

    // Indirect execution command signature
    device->dispatch_indirect = create_command_signature(
        raw_device.Get(),
        D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
        sizeof(D3D12_DISPATCH_ARGUMENTS)
    );

    // CmdCopyBufferToImage compute shader
    {
        const D3D12_DESCRIPTOR_RANGE range {
            D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            1,
            1,
            0,
            0,
        };

        D3D12_ROOT_PARAMETER parameters[3];
        CD3DX12_ROOT_PARAMETER::InitAsShaderResourceView(parameters[0], 0);
        CD3DX12_ROOT_PARAMETER::InitAsDescriptorTable(
            parameters[1],
            1,
            &range
        );
        CD3DX12_ROOT_PARAMETER::InitAsConstants(parameters[2], 3, 0);

        const D3D12_ROOT_SIGNATURE_DESC signature_desc {
            3,
            parameters,
            0,
            nullptr,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };

        ComPtr<ID3DBlob> signature_blob { nullptr };
        {
            ComPtr<ID3DBlob> error {nullptr };
            auto const hr {
                D3D12SerializeRootSignature(
                    &signature_desc,
                    D3D_ROOT_SIGNATURE_VERSION_1_0,
                    &signature_blob,
                    &error
                )
            };

            if (error) {
                ERR("D3D12SerializeRootSignature error: {}", error->GetBufferPointer());
            }
        }

        auto const hr {
            (*device)->CreateRootSignature(
                0,
                signature_blob->GetBufferPointer(),
                signature_blob->GetBufferSize(),
                IID_PPV_ARGS(&device->signature_buffer_to_image)
            )
        };

        const auto cs_buffer_to_image = compile_shader("cs_5_1", "CopyBufferToImage", copy_buffer_to_image_cs);

        const D3D12_COMPUTE_PIPELINE_STATE_DESC desc {
            device->signature_buffer_to_image.Get(),
            D3D12_SHADER_BYTECODE {
                cs_buffer_to_image->GetBufferPointer(),
                cs_buffer_to_image->GetBufferSize(),
            },
            0,
            D3D12_CACHED_PIPELINE_STATE { },
            D3D12_PIPELINE_STATE_FLAG_NONE,
        };

        {
            const auto hr {
                (*device)->CreateComputePipelineState(&desc, IID_PPV_ARGS(&device->pso_buffer_to_image))
            };

            // TODO: error
        }
    }

    // BlitImage2D compute shader
    {
        const D3D12_DESCRIPTOR_RANGE range[] = {
            D3D12_DESCRIPTOR_RANGE {
                D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                1,
                0,
                0,
                0,
            },
            D3D12_DESCRIPTOR_RANGE {
                D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                1,
                1,
                0,
                D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            },
        };

        D3D12_ROOT_PARAMETER parameters[2];
        CD3DX12_ROOT_PARAMETER::InitAsDescriptorTable(
            parameters[0],
            2,
            range
        );
        CD3DX12_ROOT_PARAMETER::InitAsConstants(parameters[1], 15, 0);

        const D3D12_STATIC_SAMPLER_DESC static_sampler {
            D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT, // TODO: nearest filter?
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            0.0,
            0,
            D3D12_COMPARISON_FUNC_ALWAYS,
            D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            0.0,
            D3D12_FLOAT32_MAX,
            0,
            0,
            D3D12_SHADER_VISIBILITY_ALL,
        };

        const D3D12_ROOT_SIGNATURE_DESC signature_desc {
            2,
            parameters,
            1,
            &static_sampler,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };

        ComPtr<ID3DBlob> signature_blob { nullptr };
        {
            ComPtr<ID3DBlob> error {nullptr };
            auto const hr {
                D3D12SerializeRootSignature(
                    &signature_desc,
                    D3D_ROOT_SIGNATURE_VERSION_1_0,
                    &signature_blob,
                    &error
                )
            };

            if (error) {
                ERR("D3D12SerializeRootSignature error: {}", error->GetBufferPointer());
            }
        }

        auto const hr {
            (*device)->CreateRootSignature(
                0,
                signature_blob->GetBufferPointer(),
                signature_blob->GetBufferSize(),
                IID_PPV_ARGS(&device->signature_blit_2d)
            )
        };

        const auto cs_blit_2d = compile_shader("cs_5_1", "BlitImage2D", blit_2d_cs);

        const D3D12_COMPUTE_PIPELINE_STATE_DESC desc {
            device->signature_blit_2d.Get(),
            D3D12_SHADER_BYTECODE {
                cs_blit_2d->GetBufferPointer(),
                cs_blit_2d->GetBufferSize(),
            },
            0,
            D3D12_CACHED_PIPELINE_STATE { },
            D3D12_PIPELINE_STATE_FLAG_NONE,
        };

        {
            const auto hr {
                (*device)->CreateComputePipelineState(&desc, IID_PPV_ARGS(&device->pso_blit_2d))
            };

            // TODO: error
        }
    }

    *pDevice = reinterpret_cast<VkDevice>(device);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(
    VkDevice                                    device,
    const VkAllocationCallbacks*                pAllocator
) {
    WARN("vkDestroyDevice unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char*                                 pLayerName,
    uint32_t*                                   pPropertyCount,
    VkExtensionProperties*                      pProperties
) {
    const std::array<VkExtensionProperties, 3> extensions {{
        {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_SURFACE_SPEC_VERSION,
        },
        {
            VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
            VK_KHR_WIN32_SURFACE_SPEC_VERSION,
        },
        { // TODO: preliminary, not fully implemented
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_SPEC_VERSION,
        },
    }};

    auto result { VK_SUCCESS };

    if (!pProperties) {
        *pPropertyCount = static_cast<uint32_t>(extensions.size());
        return VK_SUCCESS;
    }

    auto num_properties { static_cast<uint32_t>(extensions.size()) };
    if (*pPropertyCount < num_properties) {
        num_properties = *pPropertyCount;
        result = VK_INCOMPLETE;
    }

    *pPropertyCount = num_properties;
    for (auto i : range(num_properties)) {
        pProperties[i] = extensions[i];
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice                            _physicalDevice,
    const char*                                 pLayerName,
    uint32_t*                                   pPropertyCount,
    VkExtensionProperties*                      pProperties
) {
    TRACE("vkEnumerateInstanceExtensionProperties");

    auto physical_device { reinterpret_cast<physical_device_t *>(_physicalDevice) };

    std::vector<VkExtensionProperties> extensions {{
        {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_SWAPCHAIN_SPEC_VERSION,
        },
        { // TODO: preliminary, not fully implemented
            VK_KHR_MAINTENANCE1_EXTENSION_NAME,
            VK_KHR_MAINTENANCE1_SPEC_VERSION,
        },
    }};

    if (physical_device->conservative_properties) {
        extensions.push_back({
            VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME,
            VK_EXT_CONSERVATIVE_RASTERIZATION_SPEC_VERSION,
        });
    }

    auto result { VK_SUCCESS };

    if (!pProperties) {
        *pPropertyCount = static_cast<uint32_t>(extensions.size());
        return VK_SUCCESS;
    }

    auto num_properties { static_cast<uint32_t>(extensions.size()) };
    if (*pPropertyCount < num_properties) {
        num_properties = *pPropertyCount;
        result = VK_INCOMPLETE;
    }

    *pPropertyCount = num_properties;
    for (auto i : range(num_properties)) {
        pProperties[i] = extensions[i];
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t*                                   pPropertyCount,
    VkLayerProperties*                          pProperties
) {
    WARN("vkEnumerateInstanceLayerProperties unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pPropertyCount,
    VkLayerProperties*                          pProperties
) {
    WARN("vkEnumerateDeviceLayerProperties unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(
    VkDevice                                    _device,
    uint32_t                                    queueFamilyIndex,
    uint32_t                                    queueIndex,
    VkQueue*                                    pQueue
) {
    TRACE("vkGetDeviceQueue");

    auto device { reinterpret_cast<device_t *>(_device) };

    switch (queueFamilyIndex) {
        case QUEUE_FAMILY_GENERAL_PRESENT: {
            *pQueue = reinterpret_cast<VkQueue>(&device->present_queue);
        } break;
        case QUEUE_FAMILY_GENERAL: {
            *pQueue = reinterpret_cast<VkQueue>(&device->general_queues[queueIndex]);
        } break;
        case QUEUE_FAMILY_COMPUTE: {
            *pQueue = reinterpret_cast<VkQueue>(&device->compute_queues[queueIndex]);
        } break;
        case QUEUE_FAMILY_COPY: {
            *pQueue = reinterpret_cast<VkQueue>(&device->copy_queues[queueIndex]);
        } break;
        default:
            ERR("Unexpected queue family index: {}", queueFamilyIndex);
            assert(true);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(
    VkQueue                                     _queue,
    uint32_t                                    submitCount,
    const VkSubmitInfo*                         pSubmits,
    VkFence                                     _fence
) {
    TRACE("vkQueueSubmit");

    auto queue { reinterpret_cast<queue_t *>(_queue) };
    auto submits { span<const VkSubmitInfo>(pSubmits, submitCount) };

    // Reset idle fence and idle event.
    // Vulkan specification ensures us exclusive queue access.
    queue->idle_fence->Signal(0);
    ::ResetEvent(queue->idle_event);

    for (auto const& submit : submits) {
        // TODO: semaphores
        auto command_buffers { span<const VkCommandBuffer>(submit.pCommandBuffers, submit.commandBufferCount) };

        std::vector<ID3D12CommandList *> command_lists(submit.commandBufferCount, nullptr);
        for (auto i : range(submit.commandBufferCount)) {
            auto command_buffer { reinterpret_cast<command_buffer_t *>(command_buffers[i]) };
            command_lists[i] = command_buffer->command_list.Get();
        }

        (*queue)->ExecuteCommandLists(submit.commandBufferCount, command_lists.data());
    }

    if (_fence != VK_NULL_HANDLE) {
        auto fence { reinterpret_cast<fence_t *>(_fence) };
        const auto hr { (*queue)->Signal(fence->fence.Get(), 1) };
        // TODO: error handling
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(
    VkQueue                                     _queue
) {
    TRACE("vkQueueWaitIdle");

    auto queue { reinterpret_cast<queue_t *>(_queue) };
    {
        const auto hr { (*queue)->Signal(queue->idle_fence.Get(), 1) };
        // TODO: error handling
    }
    {
        const auto hr { queue->idle_fence->SetEventOnCompletion(1, queue->idle_event) };
        // TODO: error handling
    }
    {
        const auto hr { WaitForSingleObject(queue->idle_event, INFINITE) };
        // TODO: error handling
    }

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(
    VkDevice                                    _device
) {
    TRACE("vkDeviceWaitIdle");

    auto device { reinterpret_cast<device_t *>(_device) };

    // TODO: Could optimize it but not very high priority
    vkQueueWaitIdle(reinterpret_cast<VkQueue>(&device->present_queue));
    for (auto& queue : device->general_queues) {
        vkQueueWaitIdle(reinterpret_cast<VkQueue>(&queue));
    }
    for (auto& queue : device->compute_queues) {
        vkQueueWaitIdle(reinterpret_cast<VkQueue>(&queue));
    }
    for (auto& queue : device->copy_queues) {
        vkQueueWaitIdle(reinterpret_cast<VkQueue>(&queue));
    }

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(
    VkDevice                                    _device,
    const VkMemoryAllocateInfo*                 pAllocateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDeviceMemory*                             pMemory
) {
    TRACE("vkAllocateMemory");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pAllocateInfo };
    auto const& heap_property { device->heap_properties[info.memoryTypeIndex] };

    const D3D12_HEAP_DESC heap_desc = {
        info.allocationSize,
        {
            D3D12_HEAP_TYPE_CUSTOM,
            heap_property.page_property,
            heap_property.memory_pool,
            0,
            0
        },
        D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT,
        D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES // TODO: resource tier 1
    };

    ComPtr<ID3D12Heap> heap { nullptr };
    {
        const auto hr { (*device)->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap)) };
        // TODO: error handling
    }

    // Create a buffer covering the whole allocation if mappable to implement Vulkan memory mapping.
    const auto mappable { info.memoryTypeIndex != 0 };
    ComPtr<ID3D12Resource> buffer { nullptr };
    if (mappable) {
        const D3D12_RESOURCE_DESC desc {
            D3D12_RESOURCE_DIMENSION_BUFFER,
            0, // TODO: alignment?
            info.allocationSize,
            1,
            1,
            1,
            DXGI_FORMAT_UNKNOWN,
            { 1, 0 },
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            D3D12_RESOURCE_FLAG_NONE
        };

        {
            auto const hr {
                (*device)->CreatePlacedResource(
                    heap.Get(),
                    0,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&buffer)
                )
            };

            // TODO: error handling
        }
    }

    *pMemory = reinterpret_cast<VkDeviceMemory>(
        new device_memory_t { heap, buffer, info.allocationSize }
    );

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkFreeMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkFreeMemory");

    if (memory != VK_NULL_HANDLE) {
        delete reinterpret_cast<device_memory_t *>(memory);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              _memory,
    VkDeviceSize                                offset,
    VkDeviceSize                                size,
    VkMemoryMapFlags                            flags,
    void**                                      ppData
) {
    TRACE("vkMapMemory");

    auto memory { reinterpret_cast<device_memory_t *>(_memory) };

    uint8_t* ptr { nullptr };
    {
        auto const hr {
            memory->buffer->Map(
                0,
                &D3D12_RANGE { 0, 0 },
                reinterpret_cast<void **>(&ptr)
            )
        };
        // TODO: error handling
    }

    *ppData = ptr + offset;

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              _memory
) {
    TRACE("vkUnmapMemory");

    auto memory { reinterpret_cast<device_memory_t *>(_memory) };
    memory->buffer->Unmap(0, &D3D12_RANGE { 0, 0 });
}

VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(
    VkDevice                                    _device,
    uint32_t                                    memoryRangeCount,
    const VkMappedMemoryRange*                  pMemoryRanges
) {
    TRACE("vkFlushMappedMemoryRanges");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto memory_ranges { span<const VkMappedMemoryRange>(pMemoryRanges, memoryRangeCount) };

    /*
    for (auto const& range : memory_ranges) {
        auto memory { reinterpret_cast<device_memory_t *>(range.memory) };
        if (memory->buffer) {
            const auto offset { range.offset };
            const auto size { range.size == VK_WHOLE_SIZE ? memory->size - offset : range.size };

            // Map und unmap immediately!
            const auto hr { memory->buffer->Map(0, &D3D12_RANGE { 0, 0 }, nullptr) };
            // TODO: error handling
            memory->buffer->Unmap(0, &D3D12_RANGE { offset, offset + size });
        }
    }
    */

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(
    VkDevice                                    device,
    uint32_t                                    memoryRangeCount,
    const VkMappedMemoryRange*                  pMemoryRanges
) {
    WARN("vkInvalidateMappedMemoryRanges unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceMemoryCommitment(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkDeviceSize*                               pCommittedMemoryInBytes
) {
    WARN("vkGetDeviceMemoryCommitment unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(
    VkDevice                                    _device,
    VkBuffer                                    _buffer,
    VkDeviceMemory                              _memory,
    VkDeviceSize                                memoryOffset
) {
    TRACE("vkBindBufferMemory");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto buffer { reinterpret_cast<buffer_t *>(_buffer) };
    auto memory { reinterpret_cast<device_memory_t *>(_memory) };

    {
        const auto hr {
            (*device)->CreatePlacedResource(
                memory->heap.Get(),
                memoryOffset,
                &D3D12_RESOURCE_DESC {
                    D3D12_RESOURCE_DIMENSION_BUFFER,
                    buffer->memory_requirements.alignment,
                    buffer->memory_requirements.size,
                    1,
                    1,
                    1,
                    DXGI_FORMAT_UNKNOWN,
                    { 1, 0 },
                    D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    buffer->usage_flags,
                },
                D3D12_RESOURCE_STATE_COMMON, // TODO
                nullptr, // TODO
                IID_PPV_ARGS(&buffer->resource)
            )
        };
        // TODO: error
    }

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(
    VkDevice                                    _device,
    VkImage                                     _image,
    VkDeviceMemory                              _memory,
    VkDeviceSize                                memoryOffset
) {
    TRACE("vkBindImageMemory");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto image { reinterpret_cast<image_t *>(_image) };
    auto memory { reinterpret_cast<device_memory_t *>(_memory) };

    {
        const auto hr {
            (*device)->CreatePlacedResource(
                memory->heap.Get(),
                memoryOffset,
                &image->resource_desc,
                D3D12_RESOURCE_STATE_COMMON, // TODO
                nullptr, // TODO
                IID_PPV_ARGS(&(image->resource))
            )
        };
        // TODO: error
    }

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(
    VkDevice                                    device,
    VkBuffer                                    _buffer,
    VkMemoryRequirements*                       pMemoryRequirements
) {
    TRACE("vkGetBufferMemoryRequirements");

    auto buffer { reinterpret_cast<buffer_t *>(_buffer) };
    *pMemoryRequirements = buffer->memory_requirements;
}

VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(
    VkDevice                                    device,
    VkImage                                     _image,
    VkMemoryRequirements*                       pMemoryRequirements
) {
    TRACE("vkGetImageMemoryRequirements");

    auto image { reinterpret_cast<image_t *>(_image) };

    *pMemoryRequirements = {
        image->allocation_info.SizeInBytes,
        image->allocation_info.Alignment,
        0x7,
    };
}

VKAPI_ATTR void VKAPI_CALL vkGetImageSparseMemoryRequirements(
    VkDevice                                    device,
    VkImage                                     image,
    uint32_t*                                   pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements*            pSparseMemoryRequirements
) {
    WARN("vkGetImageSparseMemoryRequirements unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties(
    VkPhysicalDevice                            physicalDevice,
    VkFormat                                    format,
    VkImageType                                 type,
    VkSampleCountFlagBits                       samples,
    VkImageUsageFlags                           usage,
    VkImageTiling                               tiling,
    uint32_t*                                   pPropertyCount,
    VkSparseImageFormatProperties*              pProperties
) {
    WARN("vkGetPhysicalDeviceSparseImageFormatProperties unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueueBindSparse(
    VkQueue                                     queue,
    uint32_t                                    bindInfoCount,
    const VkBindSparseInfo*                     pBindInfo,
    VkFence                                     fence
) {
    WARN("vkQueueBindSparse unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(
    VkDevice                                    _device,
    const VkFenceCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence
) {
    TRACE("vkCreateFence");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pCreateInfo };

    ComPtr<ID3D12Fence> fence { nullptr };
    const auto initial_value = info.flags & VK_FENCE_CREATE_SIGNALED_BIT ? 1 : 0;
    const auto hr { (*device)->CreateFence(
        initial_value,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&fence)
    )};
    // TODO: error handling

    *pFence = reinterpret_cast<VkFence>(new fence_t { fence });

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyFence(
    VkDevice                                    device,
    VkFence                                     fence,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyFence");
    if (fence != VK_NULL_HANDLE) {
        delete reinterpret_cast<fence_t *>(fence);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(
    VkDevice                                    device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences
) {
    TRACE("vkResetFences");
    auto fences { span<const VkFence>(pFences, fenceCount) };

    for (auto _fence : fences) {
        auto fence { reinterpret_cast<fence_t *>(_fence) };
        fence->fence->Signal(0);
    }

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(
    VkDevice                                    device,
    VkFence                                     fence
) {
    WARN("vkGetFenceStatus unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(
    VkDevice                                    _device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences,
    VkBool32                                    waitAll,
    uint64_t                                    timeout
) {
    TRACE("vkWaitForFences");

    // TODO: synchronization guarentees of fences

    auto device { reinterpret_cast<device_t *>(_device) };

    auto event { ::CreateEvent(nullptr, FALSE, FALSE, nullptr) };

    std::vector<UINT64> fence_values(1, fenceCount);
    std::vector<ID3D12Fence*> fences;
    for (auto i : range(fenceCount)) {
        auto fence { reinterpret_cast<fence_t *>(pFences[i]) };
        fences.emplace_back(fence->fence.Get());
    }

    {
        const auto hr {
            (*device)->SetEventOnMultipleFenceCompletion(
                const_cast<ID3D12Fence* const*>(fences.data()),
                fence_values.data(),
                fenceCount,
                waitAll ? D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL : D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY,
                event
            )
        };
    }

    const auto timeout_ms { static_cast<DWORD>(saturated_add(timeout, 999'999u) / 1'000'000u) };

    const auto hr { ::WaitForSingleObject(event, timeout_ms) };
    ::CloseHandle(event);

    switch (hr) {
        case WAIT_OBJECT_0: return VK_SUCCESS;
        case WAIT_TIMEOUT: return VK_TIMEOUT;
        default: ERR("Unexpected return value on `vkWaitForFences`: {}", hr);
    }

    // TODO
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(
    VkDevice                                    _device,
    const VkSemaphoreCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSemaphore*                                pSemaphore
) {
    TRACE("vkCreateSemaphore");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pCreateInfo };

    ComPtr<ID3D12Fence> fence { nullptr };
    const auto hr { (*device)->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&fence)
    )};

    // TODO: error handling

    *pSemaphore = reinterpret_cast<VkSemaphore>(new semaphore_t { fence });

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(
    VkDevice                                    device,
    VkSemaphore                                 semaphore,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroySemaphore");
    if (semaphore != VK_NULL_HANDLE) {
        delete reinterpret_cast<semaphore_t *>(semaphore);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateEvent(
    VkDevice                                    device,
    const VkEventCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkEvent*                                    pEvent
) {
    WARN("vkCreateEvent unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyEvent(
    VkDevice                                    device,
    VkEvent                                     event,
    const VkAllocationCallbacks*                pAllocator
) {
    WARN("vkDestroyEvent unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetEventStatus(
    VkDevice                                    device,
    VkEvent                                     event
) {
    WARN("vkGetEventStatus unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkSetEvent(
    VkDevice                                    device,
    VkEvent                                     event
) {
    WARN("vkSetEvent unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetEvent(
    VkDevice                                    device,
    VkEvent                                     event
) {
    WARN("vkResetEvent unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(
    VkDevice                                    device,
    const VkQueryPoolCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkQueryPool*                                pQueryPool
) {
    WARN("vkCreateQueryPool unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(
    VkDevice                                    device,
    VkQueryPool                                 queryPool,
    const VkAllocationCallbacks*                pAllocator
) {
    WARN("vkDestroyQueryPool unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(
    VkDevice                                    device,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    size_t                                      dataSize,
    void*                                       pData,
    VkDeviceSize                                stride,
    VkQueryResultFlags                          flags
) {
    WARN("vkGetQueryPoolResults unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(
    VkDevice                                    device,
    const VkBufferCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBuffer*                                   pBuffer
) {
    TRACE("vkCreateBuffer");

    auto const& info { *pCreateInfo };

    // TODO: sparse, resident, aliasing etc.

    // Constant buffers view sizes need to be aligned (256).
    // Together with the offset alignment we can enforce an aligned CBV size.
    // without oversubscribing the buffer.
    auto size { info.size };
    if (info.usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
        size = up_align(size, 256);
    };
    if (info.usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
        size = up_align(size, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    };

    const VkMemoryRequirements memory_requirements {
        size,
        D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
        0x7
    };

    auto usage_flags { D3D12_RESOURCE_FLAG_NONE };
    if (info.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        usage_flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    // TODO: buffer clearing

    *pBuffer = reinterpret_cast<VkBuffer>(new buffer_t { nullptr, memory_requirements, usage_flags });

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyBuffer");
    if (buffer != VK_NULL_HANDLE) {
        delete reinterpret_cast<buffer_t *>(buffer);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateBufferView(
    VkDevice                                    device,
    const VkBufferViewCreateInfo*               pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBufferView*                               pView
) {
    WARN("vkCreateBufferView unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyBufferView(
    VkDevice                                    device,
    VkBufferView                                bufferView,
    const VkAllocationCallbacks*                pAllocator
) {
    WARN("vkDestroyBufferView unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(
    VkDevice                                    _device,
    const VkImageCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkImage*                                    pImage
) {
    TRACE("vkCreateImage");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pCreateInfo };

    D3D12_RESOURCE_DIMENSION dimension;
    switch (info.imageType) {
        case VK_IMAGE_TYPE_1D: dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D; break;
        case VK_IMAGE_TYPE_2D: dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; break;
        case VK_IMAGE_TYPE_3D: dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D; break;
        default: ERR("Unexpected image type {}", info.imageType); assert(true);
    }

    const auto format { formats[info.format] };

    auto flags { D3D12_RESOURCE_FLAG_NONE };
    if (info.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if (info.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    }
    if (info.usage & VK_IMAGE_USAGE_STORAGE_BIT ||
        (info.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT &&
         formats_property[info.format].optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
    {
        // Unaligned copies might need to go down the CS slow path..
        flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }

    const D3D12_RESOURCE_DESC desc {
        dimension,
        0, // TODO
        info.extent.width,
        info.extent.height,
        static_cast<UINT16>(
            dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ?
                info.extent.depth :
                info.arrayLayers
        ),
        static_cast<UINT16>(info.mipLevels),
        format,
        { info.samples, 0 }, // TODO: quality
        D3D12_TEXTURE_LAYOUT_UNKNOWN, // TODO
        flags
    };

    const auto allocation_info { (*device)->GetResourceAllocationInfo(0, 1, &desc) };

    *pImage = reinterpret_cast<VkImage>(
        new image_t {
            nullptr,
            allocation_info,
            desc,
            formats_block[info.format],
            info.usage,
        }
    );

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImage(
    VkDevice                                    device,
    VkImage                                     image,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyImage");

    if (image != VK_NULL_HANDLE) {
        delete reinterpret_cast<image_t *>(image);
    }
}

VKAPI_ATTR void VKAPI_CALL vkGetImageSubresourceLayout(
    VkDevice                                    device,
    VkImage                                     image,
    const VkImageSubresource*                   pSubresource,
    VkSubresourceLayout*                        pLayout
) {
    WARN("vkGetImageSubresourceLayout unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(
    VkDevice                                    _device,
    const VkImageViewCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkImageView*                                pView
) {
    TRACE("vkCreateImageView");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pCreateInfo };
    auto image { reinterpret_cast<image_t *>(info.image) };

    const auto format { formats[info.format] };

    // TODO: PORTABILITY: ComponentMapping? Formats compatiblity?

    auto image_view { new image_view_t { image->resource.Get() }};
    if (image->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        image_view->rtv = device->create_render_target_view(
            image,
            info.viewType,
            format,
            info.subresourceRange
        );
    }
    if (image->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        image_view->dsv = device->create_depth_stencil_view(
            image,
            info.viewType,
            format,
            info.subresourceRange
        );
    }
    if (image->usage & VK_IMAGE_USAGE_SAMPLED_BIT) {
        image_view->srv = device->create_shader_resource_view(
            image,
            info.viewType,
            format,
            info.components,
            info.subresourceRange
        );
    }
    if (image->usage & VK_IMAGE_USAGE_STORAGE_BIT) {
        image_view->uav = device->create_unordered_access_view(
            image->resource.Get(),
            info.viewType,
            format,
            info.subresourceRange
        );
    }

    *pView = reinterpret_cast<VkImageView>(image_view);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(
    VkDevice                                    _device,
    VkImageView                                 _imageView,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyImageView");

    auto device { reinterpret_cast<device_t *>(_device) };

    if (_imageView != VK_NULL_HANDLE) {
        auto image_view { reinterpret_cast<image_view_t *>(_imageView) };
        if (image_view->rtv) {
            device->destroy_render_target_view(*image_view->rtv);
        }
        if (image_view->dsv) {
            device->destroy_depth_stencil_view(*image_view->dsv);
        }
        if (image_view->srv) {
            device->destroy_shader_resource_view(*image_view->srv);
        }
        if (image_view->uav) {
            device->destroy_unordered_access_view(*image_view->uav);
        }

        delete image_view;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(
    VkDevice                                    device,
    const VkShaderModuleCreateInfo*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkShaderModule*                             pShaderModule
) {
    TRACE("vkCreateShaderModule");

    auto const& info { *pCreateInfo };

    // Copying the SPIR-V code as the final HLSL source depends on the pipeline layout.
    *pShaderModule = reinterpret_cast<VkShaderModule>(
        new shader_module_t { std::vector<uint32_t>(info.pCode, info.pCode + info.codeSize / 4) }
    );

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(
    VkDevice                                    device,
    VkShaderModule                              shaderModule,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyShaderModule");

    if (shaderModule != VK_NULL_HANDLE) {
        delete reinterpret_cast<shader_module_t *>(shaderModule);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(
    VkDevice                                    device,
    const VkPipelineCacheCreateInfo*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineCache*                            pPipelineCache
) {
    TRACE("vkCreatePipelineCache");
    WARN("vkCreatePipelineCache unimplemented");

    *pPipelineCache = reinterpret_cast<VkPipelineCache>(new pipeline_cache_t);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineCache(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyPipelineCache");
    if (pipelineCache != VK_NULL_HANDLE) {
        delete reinterpret_cast<pipeline_cache_t *>(pipelineCache);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPipelineCacheData(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    size_t*                                     pDataSize,
    void*                                       pData
) {
    WARN("vkGetPipelineCacheData unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkMergePipelineCaches(
    VkDevice                                    device,
    VkPipelineCache                             dstCache,
    uint32_t                                    srcCacheCount,
    const VkPipelineCache*                      pSrcCaches
) {
    WARN("vkMergePipelineCaches unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(
    VkDevice                                    _device,
    VkPipelineCache                             _pipelineCache,
    uint32_t                                    createInfoCount,
    const VkGraphicsPipelineCreateInfo*         pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines
) {
    TRACE("vkCreateGraphicsPipelines");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto pipeline_cache { reinterpret_cast<pipeline_cache_t *>(_pipelineCache) };
    auto infos { span<const VkGraphicsPipelineCreateInfo>(pCreateInfos, createInfoCount) };

    for (auto i : range(createInfoCount)) {
        auto const& info = infos[i];
        auto layout { reinterpret_cast<pipeline_layout_t *>(info.layout) };
        auto signature { layout->signature.Get() };
        auto render_pass { reinterpret_cast<render_pass_t *>(info.renderPass) };
        auto const& subpass { render_pass->subpasses[info.subpass] };

        auto pipeline = new pipeline_t();

        // Compile shaders
        auto stages { span<const VkPipelineShaderStageCreateInfo>(info.pStages, info.stageCount) };
        ComPtr<ID3DBlob> vertex_shader { nullptr };
        ComPtr<ID3DBlob> domain_shader { nullptr };
        ComPtr<ID3DBlob> hull_shader { nullptr };
        ComPtr<ID3DBlob> geometry_shader { nullptr };
        ComPtr<ID3DBlob> pixel_shader { nullptr };

        for (auto const& stage : stages) {
            auto [entry_name, shader_code] = translate_spirv(stage, layout);

            switch (stage.stage) {
                case VK_SHADER_STAGE_VERTEX_BIT: {
                    vertex_shader = compile_shader("vs_5_1", entry_name, shader_code);
                } break;
                case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT: {
                    domain_shader = compile_shader("ds_5_1", entry_name, shader_code);
                } break;
                case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: {
                    hull_shader = compile_shader("hs_5_1", entry_name, shader_code);
                } break;
                case VK_SHADER_STAGE_GEOMETRY_BIT: {
                    geometry_shader = compile_shader("gs_5_1", entry_name, shader_code);
                } break;
                case VK_SHADER_STAGE_FRAGMENT_BIT: {
                    pixel_shader = compile_shader("ps_5_1", entry_name, shader_code);
                } break;
            }
        }

        auto shader_bc = [] (ComPtr<ID3DBlob> shader) {
            return shader ?
                D3D12_SHADER_BYTECODE { shader->GetBufferPointer(), shader->GetBufferSize() } :
                D3D12_SHADER_BYTECODE { 0, 0 };
        };

        // These handles must be valid.
        auto const& vertex_input { *info.pVertexInputState };
        auto const& rasterization_state { *info.pRasterizationState };
        auto const& input_assembly { *info.pInputAssemblyState };

        // Dynamic states
        auto static_viewports { true };
        auto static_scissors { true };
        auto static_blend_factors { true };
        auto static_depth_bounds { true };
        auto static_stencil_reference { true };
        if (info.pDynamicState) {
            auto const& dynamic_state { *info.pDynamicState };
            auto states {
                span<const VkDynamicState>(dynamic_state.pDynamicStates, dynamic_state.dynamicStateCount)
            };

            for (auto const& state : states) {
                switch (state) {
                    case VK_DYNAMIC_STATE_DEPTH_BIAS: pipeline->dynamic_states |= DYNAMIC_STATE_DEPTH_BIAS; break;
                    case VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK: pipeline->dynamic_states |= DYNAMIC_STATE_STENCIL_COMPARE_MASK; break;
                    case VK_DYNAMIC_STATE_STENCIL_WRITE_MASK: pipeline->dynamic_states |= DYNAMIC_STATE_STENCIL_WRITE_MASK; break;

                    case VK_DYNAMIC_STATE_VIEWPORT: static_viewports = false; break;
                    case VK_DYNAMIC_STATE_SCISSOR: static_scissors = false; break;
                    case VK_DYNAMIC_STATE_BLEND_CONSTANTS: static_blend_factors = false; break;
                    case VK_DYNAMIC_STATE_DEPTH_BOUNDS: static_depth_bounds = false; break;
                    case VK_DYNAMIC_STATE_STENCIL_REFERENCE: static_stencil_reference = false; break;
                }
            }
        }

        if (static_viewports) {
            auto const& viewport_state { *info.pViewportState };
            auto vk_viewports { span<const VkViewport>(viewport_state.pViewports, viewport_state.viewportCount) };

            std::vector<D3D12_VIEWPORT> viewports;
            for (auto const& vp : vk_viewports) {
                viewports.emplace_back(
                    D3D12_VIEWPORT { vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth }
                );
            }
            pipeline->static_viewports = viewports;
        }
        if (static_scissors) {
            auto const& viewport_state { *info.pViewportState };
            auto vk_scissors { span<const VkRect2D>(viewport_state.pScissors, viewport_state.scissorCount) };

            std::vector<D3D12_RECT> scissors;
            for (auto const& scissor : vk_scissors) {
                scissors.emplace_back(
                    D3D12_RECT {
                        scissor.offset.x,
                        scissor.offset.y,
                        static_cast<LONG>(scissor.offset.x + scissor.extent.width),
                        static_cast<LONG>(scissor.offset.y + scissor.extent.height),
                    }
                );
            }
            pipeline->static_scissors = scissors;
        }

        const auto primitive_restart { input_assembly.primitiveRestartEnable == VK_TRUE };
        const auto index_strip_cut {
            primitive_restart ?
                D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF :
                D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED
        };

        if (primitive_restart) {
            pipeline->dynamic_states |= DYNAMIC_STATE_PRIMITIVE_RESTART;
        }

        // Vertex input state
        std::array<uint32_t, MAX_VERTEX_BUFFER_SLOTS> strides { 0 };
        for (auto i : range(vertex_input.vertexBindingDescriptionCount)) {
            auto const& binding { vertex_input.pVertexBindingDescriptions[i] };
            strides[binding.binding] = binding.stride;
        }

        std::vector<D3D12_INPUT_ELEMENT_DESC> input_elements {};
        for (auto attribute: range(vertex_input.vertexAttributeDescriptionCount)) {
            auto const& attribute_desc { vertex_input.pVertexAttributeDescriptions[attribute] };
            auto const& binding_desc { vertex_input.pVertexBindingDescriptions[attribute_desc.binding] };

            input_elements.emplace_back(D3D12_INPUT_ELEMENT_DESC {
                "TEXCOORD",
                attribute_desc.location,
                formats[attribute_desc.format],
                attribute_desc.binding,
                attribute_desc.offset,
                binding_desc.inputRate == VK_VERTEX_INPUT_RATE_VERTEX ?
                    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA :
                    D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                binding_desc.inputRate == VK_VERTEX_INPUT_RATE_VERTEX ? 0u : 1u,
            });
        }

        // Input assembly state && tessellation
        auto topology_type { D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED };
        switch (input_assembly.topology) {
            case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: topology_type = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT; break;
            case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
            case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
            case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
            case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
                topology_type = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE; break;
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
                topology_type = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; break;
            case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST: topology_type = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH; break;
            // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
            //   PORTABILITY: Unsupported
        }

        auto topology { D3D_PRIMITIVE_TOPOLOGY_UNDEFINED };
        switch (input_assembly.topology) {
            case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
                topology = D3D_PRIMITIVE_TOPOLOGY_POINTLIST; break;
            case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
                topology = D3D_PRIMITIVE_TOPOLOGY_LINELIST; break;
            case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
                topology = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP; break;
            case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
                topology = D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ; break;
            case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
                topology = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ; break;
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
                topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST; break;
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
                topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP; break;
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
                topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ; break;
            case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
                topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ; break;
            case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST: {
                // Must be greater than 0 and `pTessellationState` must be valid.
                const auto control_points { info.pTessellationState->patchControlPoints };
                topology = static_cast<D3D_PRIMITIVE_TOPOLOGY>(D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST + control_points - 1);
            } break;
        }

        // Color blend desc
        const auto uses_color_attachments {
            std::any_of(
                subpass.color_attachments.cbegin(),
                subpass.color_attachments.cend(),
                [] (VkAttachmentReference const& ref) {
                    return ref.attachment != VK_ATTACHMENT_UNUSED;
                }
            )
        };

        D3D12_BLEND_DESC blend_desc {
            FALSE,
            FALSE,
            { 0 }
        };

        if (!rasterization_state.rasterizerDiscardEnable && uses_color_attachments) {
            auto const& color_blend_state { *info.pColorBlendState };

            if (static_blend_factors) {
                pipeline->static_blend_factors = blend_factors_t {
                    color_blend_state.blendConstants[0],
                    color_blend_state.blendConstants[1],
                    color_blend_state.blendConstants[2],
                    color_blend_state.blendConstants[3],
                };
            }

            auto blend_factor = [] (VkBlendFactor factor) {
                switch (factor) {
                    case VK_BLEND_FACTOR_ZERO:
                        return D3D12_BLEND_ZERO;
                    case VK_BLEND_FACTOR_ONE:
                        return D3D12_BLEND_ONE;
                    case VK_BLEND_FACTOR_SRC_COLOR:
                        return D3D12_BLEND_SRC_COLOR;
                    case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
                        return D3D12_BLEND_INV_SRC_COLOR;
                    case VK_BLEND_FACTOR_DST_COLOR:
                        return D3D12_BLEND_DEST_COLOR;
                    case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR:
                        return D3D12_BLEND_INV_DEST_COLOR;
                    case VK_BLEND_FACTOR_SRC_ALPHA:
                        return D3D12_BLEND_SRC_ALPHA;
                    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
                        return D3D12_BLEND_INV_SRC_ALPHA;
                    case VK_BLEND_FACTOR_DST_ALPHA:
                        return D3D12_BLEND_DEST_ALPHA;
                    case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
                        return D3D12_BLEND_INV_DEST_ALPHA;
                    case VK_BLEND_FACTOR_CONSTANT_COLOR:
                        return D3D12_BLEND_BLEND_FACTOR;
                    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
                        return  D3D12_BLEND_INV_BLEND_FACTOR;
                    case VK_BLEND_FACTOR_CONSTANT_ALPHA:
                        return D3D12_BLEND_BLEND_FACTOR;
                    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA:
                        return D3D12_BLEND_INV_BLEND_FACTOR;
                    case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE:
                        return D3D12_BLEND_SRC_ALPHA_SAT;
                    case VK_BLEND_FACTOR_SRC1_COLOR:
                        return D3D12_BLEND_SRC1_COLOR;
                    case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR:
                        return D3D12_BLEND_INV_SRC1_COLOR;
                    case VK_BLEND_FACTOR_SRC1_ALPHA:
                        return D3D12_BLEND_SRC1_ALPHA;
                    case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA:
                        return D3D12_BLEND_INV_SRC1_ALPHA;
                }
                return D3D12_BLEND_ZERO;
            };

            auto blend_op = [] (VkBlendOp op) {
                switch (op) {
                    case VK_BLEND_OP_ADD: return D3D12_BLEND_OP_ADD;
                    case VK_BLEND_OP_SUBTRACT: return D3D12_BLEND_OP_SUBTRACT;
                    case VK_BLEND_OP_REVERSE_SUBTRACT: return D3D12_BLEND_OP_REV_SUBTRACT;
                    case VK_BLEND_OP_MIN: return D3D12_BLEND_OP_MIN;
                    case VK_BLEND_OP_MAX: return D3D12_BLEND_OP_MAX;
                }
                return D3D12_BLEND_OP_ADD;
            };

            // TODO: we could enable this when the feature is enabled
            blend_desc.IndependentBlendEnable = TRUE; // Vulkan always enables this
            if (color_blend_state.attachmentCount) {
                auto blend_attachments_states {
                    span<const VkPipelineColorBlendAttachmentState>(color_blend_state.pAttachments, color_blend_state.attachmentCount)
                };

                D3D12_LOGIC_OP logic_op { D3D12_LOGIC_OP_CLEAR };
                switch (color_blend_state.logicOp) {
                    case VK_LOGIC_OP_CLEAR:         logic_op = D3D12_LOGIC_OP_CLEAR; break;
                    case VK_LOGIC_OP_AND:           logic_op = D3D12_LOGIC_OP_AND; break;
                    case VK_LOGIC_OP_AND_REVERSE:   logic_op = D3D12_LOGIC_OP_AND_REVERSE; break;
                    case VK_LOGIC_OP_COPY:          logic_op = D3D12_LOGIC_OP_COPY; break;
                    case VK_LOGIC_OP_AND_INVERTED:  logic_op = D3D12_LOGIC_OP_AND_INVERTED; break;
                    case VK_LOGIC_OP_NO_OP:         logic_op = D3D12_LOGIC_OP_NOOP; break;
                    case VK_LOGIC_OP_XOR:           logic_op = D3D12_LOGIC_OP_XOR; break;
                    case VK_LOGIC_OP_OR:            logic_op = D3D12_LOGIC_OP_OR; break;
                    case VK_LOGIC_OP_NOR:           logic_op = D3D12_LOGIC_OP_NOR; break;
                    case VK_LOGIC_OP_EQUIVALENT:    logic_op = D3D12_LOGIC_OP_EQUIV; break;
                    case VK_LOGIC_OP_INVERT:        logic_op = D3D12_LOGIC_OP_INVERT; break;
                    case VK_LOGIC_OP_OR_REVERSE:    logic_op = D3D12_LOGIC_OP_OR_REVERSE; break;
                    case VK_LOGIC_OP_COPY_INVERTED: logic_op = D3D12_LOGIC_OP_COPY_INVERTED; break;
                    case VK_LOGIC_OP_OR_INVERTED:   logic_op = D3D12_LOGIC_OP_OR_INVERTED; break;
                    case VK_LOGIC_OP_NAND:          logic_op = D3D12_LOGIC_OP_NAND; break;
                    case VK_LOGIC_OP_SET:           logic_op = D3D12_LOGIC_OP_SET; break;
                }

                for (auto blend : range(blend_attachments_states.size())) {
                    auto const& attachment_state { blend_attachments_states[blend] };
                    blend_desc.RenderTarget[blend] = {
                        attachment_state.blendEnable == VK_TRUE,
                        color_blend_state.logicOpEnable == VK_TRUE,
                        blend_factor(attachment_state.srcColorBlendFactor),
                        blend_factor(attachment_state.dstColorBlendFactor),
                        blend_op(attachment_state.colorBlendOp),
                        blend_factor(attachment_state.srcAlphaBlendFactor),
                        blend_factor(attachment_state.dstAlphaBlendFactor),
                        blend_op(attachment_state.alphaBlendOp),
                        logic_op,
                        static_cast<UINT8>(attachment_state.colorWriteMask) // Same values as D3D12
                    };
                }
            }
        }

        // rasterizer desc
        auto fill_mode { D3D12_FILL_MODE_SOLID };
        switch (rasterization_state.polygonMode) {
            case VK_POLYGON_MODE_FILL: fill_mode = D3D12_FILL_MODE_SOLID; break;
            case VK_POLYGON_MODE_LINE: fill_mode = D3D12_FILL_MODE_WIREFRAME; break;
            // VK_POLYGON_MODE_POINT: // PORTABILITY: not supported on d3d12
        }

        auto cull_mode { D3D12_CULL_MODE_NONE };
        switch (rasterization_state.cullMode) {
            case VK_CULL_MODE_NONE: cull_mode = D3D12_CULL_MODE_NONE; break;
            case VK_CULL_MODE_FRONT_BIT: cull_mode = D3D12_CULL_MODE_FRONT; break;
            case VK_CULL_MODE_BACK_BIT: cull_mode = D3D12_CULL_MODE_BACK; break;
            // VK_CULL_MODE_FRONT_AND_BACK:
            //     PORTABILITY: not supported on d3d12, would clash with pipeline statistics
            //                  by emulating it via rasterization discard
        }

        const auto depth_bias { rasterization_state.depthBiasEnable };

        D3D12_RASTERIZER_DESC rasterizer_desc {
            fill_mode,
            cull_mode,
            rasterization_state.frontFace == VK_FRONT_FACE_COUNTER_CLOCKWISE ? TRUE : FALSE, // TODO: double check
            depth_bias ? static_cast<INT>(rasterization_state.depthBiasConstantFactor) : 0,
            depth_bias ? rasterization_state.depthBiasClamp : 0.0f,
            depth_bias ? rasterization_state.depthBiasSlopeFactor : 0.0f,
            !rasterization_state.depthClampEnable,
            FALSE,
            FALSE, // TODO: AA lines
            0, // TODO: forced sample count
            D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
        };

        pipeline->static_state.depth_bias = rasterizer_desc.DepthBias;
        pipeline->static_state.depth_bias_clamp = rasterizer_desc.DepthBiasClamp;
        pipeline->static_state.depth_bias_slope = rasterizer_desc.SlopeScaledDepthBias;

        // multi sampling
        UINT sample_mask { ~0u };
        DXGI_SAMPLE_DESC sample_desc { 1, 0 };
        if (!rasterization_state.rasterizerDiscardEnable) {
            auto const& multisampling_state { *info.pMultisampleState };

            rasterizer_desc.MultisampleEnable = multisampling_state.rasterizationSamples != VK_SAMPLE_COUNT_1_BIT;
            sample_desc.Count = multisampling_state.rasterizationSamples;
            if (multisampling_state.pSampleMask) {
                sample_mask = *multisampling_state.pSampleMask;
            }

            blend_desc.AlphaToCoverageEnable = multisampling_state.alphaToCoverageEnable;
            // TODO: alpha to one
        }

        // depth stencil desc
        const auto depth_attachment { subpass.depth_attachment.attachment };
        const auto dsv_format {
            depth_attachment == VK_ATTACHMENT_UNUSED ?
                DXGI_FORMAT_UNKNOWN :
                formats[render_pass->attachments[depth_attachment].desc.format]
        };

        const auto uses_depth_stencil_attachment { depth_attachment != VK_ATTACHMENT_UNUSED };
        D3D12_DEPTH_STENCIL_DESC depth_stencil_desc { };

        auto stencil_op = [=] (VkStencilOp vk_op) {
            switch (vk_op) {
                case VK_STENCIL_OP_KEEP: return D3D12_STENCIL_OP_KEEP;
                case VK_STENCIL_OP_ZERO: return D3D12_STENCIL_OP_ZERO;
                case VK_STENCIL_OP_REPLACE: return D3D12_STENCIL_OP_REPLACE;
                case VK_STENCIL_OP_INCREMENT_AND_CLAMP: return D3D12_STENCIL_OP_INCR_SAT;
                case VK_STENCIL_OP_DECREMENT_AND_CLAMP: return D3D12_STENCIL_OP_DECR_SAT;
                case VK_STENCIL_OP_INVERT: return D3D12_STENCIL_OP_INVERT;
                case VK_STENCIL_OP_INCREMENT_AND_WRAP: return D3D12_STENCIL_OP_INCR;
                case VK_STENCIL_OP_DECREMENT_AND_WRAP: return D3D12_STENCIL_OP_DECR;

                default: return D3D12_STENCIL_OP_KEEP;
            }
        };

        auto stencil_op_state = [&] (VkStencilOpState const& state) {
            return D3D12_DEPTH_STENCILOP_DESC {
                stencil_op(state.failOp),
                stencil_op(state.depthFailOp),
                stencil_op(state.passOp),
                compare_op(state.compareOp)
            };
        };

        if (!rasterization_state.rasterizerDiscardEnable && uses_depth_stencil_attachment) {
            auto const& depth_stencil_state { *info.pDepthStencilState };

            if (depth_stencil_state.depthBoundsTestEnable == VK_TRUE && static_depth_bounds) {
                pipeline->static_depth_bounds = std::make_tuple(
                    depth_stencil_state.minDepthBounds,
                    depth_stencil_state.maxDepthBounds
                );
            }
            if (static_stencil_reference) {
                pipeline->static_stencil_reference = depth_stencil_state.front.reference; // PORTABILITY: front == back
            }

            depth_stencil_desc.DepthEnable = depth_stencil_state.depthTestEnable == VK_TRUE;
            depth_stencil_desc.DepthWriteMask = depth_stencil_state.depthWriteEnable ?
                D3D12_DEPTH_WRITE_MASK_ALL :
                D3D12_DEPTH_WRITE_MASK_ZERO;
            depth_stencil_desc.DepthFunc = compare_op(depth_stencil_state.depthCompareOp);
            depth_stencil_desc.StencilEnable = depth_stencil_state.stencilTestEnable == VK_TRUE;
            depth_stencil_desc.StencilReadMask = depth_stencil_state.front.compareMask; // PORTABILITY: front == back
            depth_stencil_desc.StencilWriteMask = depth_stencil_state.front.writeMask; // PORTABILITY: front == back
            depth_stencil_desc.FrontFace = stencil_op_state(depth_stencil_state.front);
            depth_stencil_desc.BackFace = stencil_op_state(depth_stencil_state.back);
        }

        pipeline->static_state.depth_bias = rasterizer_desc.DepthBias;
        pipeline->static_state.depth_bias_clamp = rasterizer_desc.DepthBiasClamp;
        pipeline->static_state.depth_bias_slope = rasterizer_desc.SlopeScaledDepthBias;
        pipeline->static_state.stencil_read_mask = depth_stencil_desc.StencilReadMask;
        pipeline->static_state.stencil_write_mask = depth_stencil_desc.StencilWriteMask;
        pipeline->static_state.strip_cut = index_strip_cut;

        // Indicate if we have a truly dynamic pso
        const auto dynamic_pso { pipeline->dynamic_states != 0 };

        const auto num_render_targets { subpass.color_attachments.size() };

        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc {
            signature,
            shader_bc(vertex_shader),
            shader_bc(pixel_shader),
            shader_bc(domain_shader),
            shader_bc(hull_shader),
            shader_bc(geometry_shader),
            D3D12_STREAM_OUTPUT_DESC { // not used
                nullptr,
                0u,
                nullptr,
                0u,
                0u,
            },
            blend_desc,
            sample_mask,
            rasterizer_desc,
            depth_stencil_desc,
            D3D12_INPUT_LAYOUT_DESC {
                input_elements.data(),
                static_cast<UINT>(input_elements.size())
            },
            index_strip_cut,
            topology_type,
            num_render_targets,
            { }, // Fill in RTV formats below
            dsv_format,
            sample_desc,
            0, // NodeMask
            D3D12_CACHED_PIPELINE_STATE { }, // TODO
            D3D12_PIPELINE_STATE_FLAG_NONE, // TODO
        };

         // RTV formats
        for (auto i : range(num_render_targets)) {
            const auto attachment { subpass.color_attachments[i].attachment };
            desc.RTVFormats[i] = attachment != VK_ATTACHMENT_UNUSED ?
                formats[render_pass->attachments[attachment].desc.format] :
                DXGI_FORMAT_UNKNOWN;
        }

        if (!dynamic_pso) {
            ComPtr<ID3D12PipelineState> pso { nullptr };
            auto const hr { (*device)->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)) };
            // TODO: error handling

            pipeline->pso = pipeline_t::unique_pso_t { pso };
        } else {
            pipeline->pso = pipeline_t::dynamic_pso_t {
                {},
                vertex_shader,
                domain_shader,
                hull_shader,
                geometry_shader,
                pixel_shader,
                desc.BlendState,
                desc.SampleMask,
                desc.RasterizerState,
                desc.DepthStencilState,
                input_elements,
                desc.NumRenderTargets,
                desc.RTVFormats[0], desc.RTVFormats[1], desc.RTVFormats[2], desc.RTVFormats[3],
                desc.RTVFormats[4], desc.RTVFormats[5], desc.RTVFormats[6], desc.RTVFormats[7],
                desc.DSVFormat,
                desc.SampleDesc,
                topology_type,
            };
        }

        pipeline->signature = signature;
        pipeline->num_signature_entries = layout->num_signature_entries;
        pipeline->root_constants = layout->root_constants;
        pipeline->num_root_constants = layout->num_root_constants;
        pipeline->vertex_strides = strides;
        pipeline->topology = topology;
        pPipelines[i] = reinterpret_cast<VkPipeline>(pipeline);
    }

    //

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(
    VkDevice                                    _device,
    VkPipelineCache                             _pipelineCache,
    uint32_t                                    createInfoCount,
    const VkComputePipelineCreateInfo*          pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines
) {
    TRACE("vkCreateComputePipelines");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto pipeline_cache { reinterpret_cast<pipeline_cache_t *>(_pipelineCache) };
    auto infos { span<const VkComputePipelineCreateInfo>(pCreateInfos, createInfoCount) };

    for (auto i : range(createInfoCount)) {
        auto const& info = infos[i];
        auto layout { reinterpret_cast<pipeline_layout_t *>(info.layout) };
        auto signature { layout->signature.Get() };

        auto [entry_name, shader_code] = translate_spirv(info.stage, layout);
        auto compute_shader = compile_shader("cs_5_1", entry_name, shader_code);

        const D3D12_SHADER_BYTECODE cs {
            compute_shader->GetBufferPointer(),
            compute_shader->GetBufferSize(),
        };

        const D3D12_COMPUTE_PIPELINE_STATE_DESC desc {
            signature,
            cs,
            0,
            D3D12_CACHED_PIPELINE_STATE { },
            D3D12_PIPELINE_STATE_FLAG_NONE,
        };

        auto pipeline = new pipeline_t;
        ComPtr<ID3D12PipelineState> pso { nullptr };
        auto const hr { (*device)->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)) };
        // TODO: errror handling

        pipeline->pso = pipeline_t::unique_pso_t { pso };

        pipeline->signature = signature;
        pipeline->num_signature_entries = layout->num_signature_entries;
        pipeline->root_constants = layout->root_constants;
        pipeline->num_root_constants = layout->num_root_constants;
        pPipelines[i] = reinterpret_cast<VkPipeline>(pipeline);
    }

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyPipeline");
    if (pipeline != VK_NULL_HANDLE) {
        delete reinterpret_cast<pipeline_t *>(pipeline);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(
    VkDevice                                    _device,
    const VkPipelineLayoutCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineLayout*                           pPipelineLayout
) {
    TRACE("vkCreatePipelineLayout");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pCreateInfo };
    auto set_layouts { span<const VkDescriptorSetLayout>(info.pSetLayouts, info.setLayoutCount) };
    auto push_constants { span<const VkPushConstantRange>(info.pPushConstantRanges, info.pushConstantRangeCount) };

    std::vector<uint32_t> tables;
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    std::vector<VkPushConstantRange> root_constants;

    size_t num_root_constants { 0 };

    // TODO: missing overlap support: merge/split push constants
    for (auto const& push_constant : push_constants) {
        D3D12_ROOT_PARAMETER parameter {
            D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            { },
            D3D12_SHADER_VISIBILITY_ALL // TODO
        };
        parameter.Constants = D3D12_ROOT_CONSTANTS {
            push_constant.offset / 4,
            PUSH_CONSTANT_REGISTER_SPACE,
            push_constant.size / 4
        };

        parameters.push_back(parameter);
        root_constants.push_back(push_constant);
        num_root_constants += push_constant.size / 4;
    }

    auto num_descriptor_ranges { 0 };
    for (auto const& _set : set_layouts) {
        auto set { reinterpret_cast<descriptor_set_layout_t *>(_set) };
        for (auto const& layout : set->layouts) {
            num_descriptor_ranges += layout.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? 2 : 1;
        }
    };

    std::vector<D3D12_DESCRIPTOR_RANGE> descriptor_ranges;
    descriptor_ranges.reserve(num_descriptor_ranges);

    // Set layouts will be translated into descriptor tables.
    // Due to the limitations of splitting samplers from  other descriptor types,
    // we might need to use 2 tables.
    for (auto i : range(info.setLayoutCount)) {
        auto set_layout { reinterpret_cast<descriptor_set_layout_t *>(set_layouts[i]) };

        uint32_t table_type = 0;

        auto descriptor_range = [] (descriptor_set_layout_t::binding_t const& binding, uint32_t space, bool is_sampler) {
            auto type { D3D12_DESCRIPTOR_RANGE_TYPE_SRV };
            switch (binding.type) {
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: type = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; break;
                case VK_DESCRIPTOR_TYPE_SAMPLER: type = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER; break;
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                    type = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: type = D3D12_DESCRIPTOR_RANGE_TYPE_CBV; break;
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                    type = is_sampler ?
                        D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER :
                        D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                } break;
                default: ERR("Unsupported descriptor type: {}", binding.type);
            }

            return D3D12_DESCRIPTOR_RANGE {
                type,
                binding.descriptor_count,
                binding.binding,
                space + DESCRIPTOR_TABLE_INITIAL_SPACE,
                D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            };
        };

        D3D12_ROOT_PARAMETER paramter {
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            { },
            D3D12_SHADER_VISIBILITY_ALL // TODO
        };

        const auto cbv_srv_uav_start { descriptor_ranges.size() };
        for (auto const& binding : set_layout->layouts) {
            if (binding.type == VK_DESCRIPTOR_TYPE_SAMPLER) {
                continue;
            }
            descriptor_ranges.emplace_back(descriptor_range(binding, 2*i, false));
        }
        const auto cbv_srv_uav_end { descriptor_ranges.size() };

        if (cbv_srv_uav_end > cbv_srv_uav_start) {
            paramter.DescriptorTable = D3D12_ROOT_DESCRIPTOR_TABLE {
                static_cast<UINT>(cbv_srv_uav_end - cbv_srv_uav_start),
                descriptor_ranges.data() + cbv_srv_uav_start
            };
            table_type |= TABLE_CBV_SRV_UAV;
            parameters.push_back(paramter);
        }

        const auto sampler_start { descriptor_ranges.size() };
        for (auto const& binding : set_layout->layouts) {
            if (binding.type != VK_DESCRIPTOR_TYPE_SAMPLER &&
                binding.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            {
                continue;
            }
            descriptor_ranges.emplace_back(descriptor_range(binding, 2*i + 1, true));
        }
        const auto sampler_end { descriptor_ranges.size() };

        if (sampler_end > sampler_start) {
            paramter.DescriptorTable = D3D12_ROOT_DESCRIPTOR_TABLE {
                static_cast<UINT>(sampler_end - sampler_start),
                descriptor_ranges.data() + sampler_start
            };
            table_type |= TABLE_SAMPLER;
            parameters.push_back(paramter);
        }

        tables.push_back(table_type);
    }

    // TODO
    const D3D12_ROOT_SIGNATURE_DESC signature_desc {
        static_cast<UINT>(parameters.size()), // num parameters
        parameters.data(), // parameters
        0u, // static samplers
        nullptr, // static samplers
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };

    ComPtr<ID3DBlob> signature_blob { nullptr };
    {
        ComPtr<ID3DBlob> error {nullptr };
        auto const hr {
            D3D12SerializeRootSignature(
                &signature_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &signature_blob,
                &error
            )
        };

        if (error) {
            ERR("D3D12SerializeRootSignature error: {}", error->GetBufferPointer());
        }
    }

    ComPtr<ID3D12RootSignature> root_signature { nullptr };

    auto const hr {
        (*device)->CreateRootSignature(
            0,
            signature_blob->GetBufferPointer(),
            signature_blob->GetBufferSize(),
            IID_PPV_ARGS(&root_signature)
        )
    };

    *pPipelineLayout = reinterpret_cast<VkPipelineLayout>(
        new pipeline_layout_t {
            root_signature,
            tables,
            root_constants,
            num_root_constants,
            parameters.size()
        }
    );

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(
    VkDevice                                    device,
    VkPipelineLayout                            pipelineLayout,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyPipelineLayout");
    if (pipelineLayout != VK_NULL_HANDLE) {
        delete reinterpret_cast<pipeline_layout_t *>(pipelineLayout);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(
    VkDevice                                    _device,
    const VkSamplerCreateInfo*                  pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSampler*                                  pSampler
) {
    TRACE("vkCreateSampler");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pCreateInfo };

    auto base_filter = [] (VkFilter vk_filter) {
        D3D12_FILTER_TYPE filter { D3D12_FILTER_TYPE_POINT };
        switch (vk_filter) {
            case VK_FILTER_NEAREST: filter = D3D12_FILTER_TYPE_POINT; break;
            case VK_FILTER_LINEAR: filter = D3D12_FILTER_TYPE_LINEAR; break;
        }
        return filter;
    };

    auto mip_filter = [] (VkSamplerMipmapMode mip) {
        D3D12_FILTER_TYPE filter { D3D12_FILTER_TYPE_POINT };
        switch (mip) {
            case VK_SAMPLER_MIPMAP_MODE_NEAREST: filter = D3D12_FILTER_TYPE_POINT; break;
            case VK_SAMPLER_MIPMAP_MODE_LINEAR: filter = D3D12_FILTER_TYPE_LINEAR; break;
        }
        return filter;
    };

    auto address_mode = [] (VkSamplerAddressMode vk_mode) {
        D3D12_TEXTURE_ADDRESS_MODE mode { D3D12_TEXTURE_ADDRESS_MODE_WRAP };
        switch (vk_mode) {
            case VK_SAMPLER_ADDRESS_MODE_REPEAT: mode = D3D12_TEXTURE_ADDRESS_MODE_WRAP; break;
            case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: mode = D3D12_TEXTURE_ADDRESS_MODE_MIRROR; break;
            case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: mode = D3D12_TEXTURE_ADDRESS_MODE_CLAMP; break;
            case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: mode = D3D12_TEXTURE_ADDRESS_MODE_BORDER; break;
            case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: mode = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE; break;
        }
        return mode;
    };

    const auto reduction {
        info.compareEnable ?
            D3D12_FILTER_REDUCTION_TYPE_COMPARISON :
            D3D12_FILTER_REDUCTION_TYPE_STANDARD
    };

    std::array<float, 4> border_color { 0.0, 0.0, 0.0, 0.0 };
    switch (info.borderColor) {
        case VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK:
        case VK_BORDER_COLOR_INT_TRANSPARENT_BLACK: {
            border_color = { 0.0, 0.0, 0.0, 0.0 };
        } break;
        case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
        case VK_BORDER_COLOR_INT_OPAQUE_BLACK: {
            border_color = { 0.0, 0.0, 0.0, 1.0 };
        } break;
        case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
        case VK_BORDER_COLOR_INT_OPAQUE_WHITE: {
            border_color = { 1.0, 1.0, 1.0, 1.0 };
        } break;
    }

    auto filter {
        D3D12_ENCODE_BASIC_FILTER(
            base_filter(info.minFilter),
            base_filter(info.magFilter),
            mip_filter(info.mipmapMode),
            reduction
        )
    };
    if (info.anisotropyEnable) {
        filter = static_cast<D3D12_FILTER>(filter | D3D12_ANISOTROPIC_FILTERING_BIT);
    }

    D3D12_SAMPLER_DESC desc {
        filter,
        address_mode(info.addressModeU),
        address_mode(info.addressModeV),
        address_mode(info.addressModeW),
        info.mipLodBias,
        static_cast<UINT>(info.maxAnisotropy),
        compare_op(info.compareOp),
        border_color[0], border_color[1], border_color[2], border_color[3],
        info.minLod,
        info.maxLod,
    };
    // TODO: unnormalized coordinates

    auto sampler = new sampler_t;
    sampler->sampler = device->descriptors_cpu_sampler.alloc();
    (*device)->CreateSampler(&desc, std::get<0>(sampler->sampler));

    *pSampler = reinterpret_cast<VkSampler>(sampler);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySampler(
    VkDevice                                    device,
    VkSampler                                   sampler,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroySampler");
    if (sampler != VK_NULL_HANDLE) {
        delete reinterpret_cast<sampler_t *>(sampler);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorSetLayout*                      pSetLayout
) {
    TRACE("vkCreateDescriptorSetLayout");

    auto const& info { *pCreateInfo };
    auto bindings { span<const VkDescriptorSetLayoutBinding>(info.pBindings, info.bindingCount) };

    auto layout = new descriptor_set_layout_t;
    for (auto const& binding : bindings) {
        layout->layouts.emplace_back(descriptor_set_layout_t::binding_t {
            binding.binding,
            binding.descriptorType,
            binding.descriptorCount,
            binding.stageFlags,
            std::vector<VkSampler>() // TODO
        });
    }

    *pSetLayout = reinterpret_cast<VkDescriptorSetLayout>(layout);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(
    VkDevice                                    device,
    VkDescriptorSetLayout                       descriptorSetLayout,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyDescriptorSetLayout");
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        delete reinterpret_cast<descriptor_set_layout_t *>(descriptorSetLayout);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(
    VkDevice                                    _device,
    const VkDescriptorPoolCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorPool*                           pDescriptorPool
) {
    TRACE("vkCreateDescriptorPool");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pCreateInfo };
    auto pool_sizes { span<const VkDescriptorPoolSize>(info.pPoolSizes, info.poolSizeCount) };

    auto num_cbv_srv_uav { 0u };
    auto num_sampler { 0u };

    for (auto const& pool_size : pool_sizes) {
        switch (pool_size.type) {
            case VK_DESCRIPTOR_TYPE_SAMPLER: {
                num_sampler += pool_size.descriptorCount;
            } break;
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                num_cbv_srv_uav += pool_size.descriptorCount;
                num_sampler += pool_size.descriptorCount;
            } break;
            default: {
                num_cbv_srv_uav += pool_size.descriptorCount;
            } break;
        }
    }

    auto pool = new descriptor_pool_t(num_cbv_srv_uav, num_sampler);

    pool->slice_cbv_srv_uav.start = device->descriptors_gpu_cbv_srv_uav.alloc(num_cbv_srv_uav);
    pool->slice_cbv_srv_uav.handle_size = device->descriptors_gpu_cbv_srv_uav.handle_size();

    pool->slice_sampler.start = device->descriptors_gpu_sampler.alloc(num_sampler);
    pool->slice_sampler.handle_size = device->descriptors_gpu_sampler.handle_size();

    *pDescriptorPool = reinterpret_cast<VkDescriptorPool>(pool);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    const VkAllocationCallbacks*                pAllocator
) {
    WARN("vkDestroyDescriptorPool unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    VkDescriptorPoolResetFlags                  flags
) {
    WARN("vkResetDescriptorPool unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(
    VkDevice                                    _device,
    const VkDescriptorSetAllocateInfo*          pAllocateInfo,
    VkDescriptorSet*                            pDescriptorSets
) {
    TRACE("vkAllocateDescriptorSets");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pAllocateInfo };
    auto pool { reinterpret_cast<descriptor_pool_t *>(info.descriptorPool) };
    auto layouts { span<const VkDescriptorSetLayout>(info.pSetLayouts, info.descriptorSetCount) };

    const auto handle_size_cbv_srv_uav { device->descriptors_gpu_cbv_srv_uav.handle_size() };
    const auto handle_size_sampler { device->descriptors_gpu_sampler.handle_size() };

    for (auto i : range(info.descriptorSetCount)) {
        auto layout { reinterpret_cast<descriptor_set_layout_t *>(layouts[i]) };
        auto descriptor_set = new descriptor_set_t;

        // TODO: precompute once
        auto num_cbv_srv_uav { 0 };
        auto num_sampler { 0 };
        for (auto const& binding : layout->layouts) {
            switch (binding.type) {
                case VK_DESCRIPTOR_TYPE_SAMPLER: {
                    num_sampler += binding.descriptor_count;
                } break;
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                    num_sampler += binding.descriptor_count;
                    num_cbv_srv_uav += binding.descriptor_count;
                } break;
                default: {
                    num_cbv_srv_uav += binding.descriptor_count;
                } break;
            }
        }

        auto [descriptor_cbv_srv_uav, descriptor_sampler] = pool->alloc(num_cbv_srv_uav, num_sampler);
        auto [descriptor_cpu_cbv_srv_uav, descriptor_gpu_cbv_srv_uav] = descriptor_cbv_srv_uav;
        auto [descriptor_cpu_sampler, descriptor_gpu_sampler] = descriptor_sampler;

        descriptor_set->start_cbv_srv_uav = num_cbv_srv_uav ?
            std::optional<D3D12_GPU_DESCRIPTOR_HANDLE>(descriptor_gpu_cbv_srv_uav) :
            std::nullopt;
        descriptor_set->start_sampler = num_sampler ?
            std::optional<D3D12_GPU_DESCRIPTOR_HANDLE>(descriptor_gpu_sampler) :
            std::nullopt;

        for (auto const& binding : layout->layouts) {
            descriptor_set->bindings.emplace(
                binding.binding,
                descriptor_set_t::binding_t {
                    descriptor_cpu_cbv_srv_uav,
                    descriptor_cpu_sampler,
                    binding.descriptor_count
                }
            );

            // advance
            switch (binding.type) {
                case VK_DESCRIPTOR_TYPE_SAMPLER: {
                    descriptor_cpu_sampler.ptr += binding.descriptor_count * handle_size_sampler;
                } break;
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                    descriptor_cpu_sampler.ptr += binding.descriptor_count * handle_size_sampler;
                    descriptor_cpu_cbv_srv_uav.ptr += binding.descriptor_count * handle_size_cbv_srv_uav;
                } break;
                default: {
                    descriptor_cpu_cbv_srv_uav.ptr += binding.descriptor_count * handle_size_cbv_srv_uav;
                } break;
            }
        }

        pDescriptorSets[i] = reinterpret_cast<VkDescriptorSet>(descriptor_set);
    }

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkFreeDescriptorSets(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet*                      pDescriptorSets
) {
    WARN("vkFreeDescriptorSets unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(
    VkDevice                                    _device,
    uint32_t                                    descriptorWriteCount,
    const VkWriteDescriptorSet*                 pDescriptorWrites,
    uint32_t                                    descriptorCopyCount,
    const VkCopyDescriptorSet*                  pDescriptorCopies
) {
    TRACE("vkUpdateDescriptorSets");

    auto device { reinterpret_cast<device_t *>(_device) };
    const auto handle_size_cbv_srv_uav { device->descriptors_gpu_cbv_srv_uav.handle_size() };
    const auto handle_size_sampler { device->descriptors_gpu_sampler.handle_size() };

    // Temporary handles for uniform and storage buffers
    std::vector<std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, heap_index_t>> temporary_handles;

    // Writes first
    auto write_descriptors_set = [&] (
        D3D12_DESCRIPTOR_HEAP_TYPE ty,
        std::function<void (
            VkWriteDescriptorSet const&,
            std::vector<D3D12_CPU_DESCRIPTOR_HANDLE>&, // dst
            std::vector<D3D12_CPU_DESCRIPTOR_HANDLE>&, // src
            std::vector<UINT>&, // dst
            std::vector<UINT>& // src
        )> const& collect
    ) {
        std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> dst_descriptor_range_starts {};
        std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> src_descriptor_range_starts {};

        std::vector<UINT> dst_descriptor_range_sizes {};
        std::vector<UINT> src_descriptor_range_sizes {};

        for (auto i : range(descriptorWriteCount)) {
            auto const& write { pDescriptorWrites[i] };

            collect(
                write,
                dst_descriptor_range_starts,
                src_descriptor_range_starts,
                dst_descriptor_range_sizes,
                src_descriptor_range_sizes
            );
        }

        if (!dst_descriptor_range_starts.empty()) {
            (*device)->CopyDescriptors(
                static_cast<UINT>(dst_descriptor_range_starts.size()),
                dst_descriptor_range_starts.data(),
                dst_descriptor_range_sizes.data(),
                static_cast<UINT>(src_descriptor_range_starts.size()),
                src_descriptor_range_starts.data(),
                src_descriptor_range_sizes.data(),
                ty
            );
        }
    };

    write_descriptors_set(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        [&] (VkWriteDescriptorSet const& write,
            std::vector<D3D12_CPU_DESCRIPTOR_HANDLE>& dst_range_starts,
            std::vector<D3D12_CPU_DESCRIPTOR_HANDLE>& src_range_starts,
            std::vector<UINT>& dst_range_sizes,
            std::vector<UINT>& src_range_sizes
        ) {
            auto descriptor_set { reinterpret_cast<descriptor_set_t *>(write.dstSet) };
            auto binding { descriptor_set->bindings.find(write.dstBinding) };
            auto array_elem { write.dstArrayElement };

            switch (write.descriptorType) {
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                    auto image_views { span<const VkDescriptorImageInfo>(write.pImageInfo, write.descriptorCount) };
                    for (auto i : range(write.descriptorCount)) {
                        auto const& view { image_views[i] };
                        auto image_view { reinterpret_cast<image_view_t *>(view.imageView) };

                        if (array_elem >= binding->second.num_descriptors) {
                            ++binding; // go to next binding slot
                        }

                        src_range_starts.push_back(std::get<0>(*image_view->srv));
                        src_range_sizes.push_back(1);

                        const auto dst_handle {
                            D3D12_CPU_DESCRIPTOR_HANDLE {
                                binding->second.start_cbv_srv_uav.ptr + array_elem * handle_size_cbv_srv_uav
                            }
                        };
                        dst_range_starts.push_back(dst_handle);
                        dst_range_sizes.push_back(1); // TODO: could batch a bit but minor.., else remove these ranges

                        array_elem += 1;
                    }
                } break;
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: {
                    auto image_views { span<const VkDescriptorImageInfo>(write.pImageInfo, write.descriptorCount) };
                    for (auto i : range(write.descriptorCount)) {
                        auto const& view { image_views[i] };
                        auto image_view { reinterpret_cast<image_view_t *>(view.imageView) };

                        if (array_elem >= binding->second.num_descriptors) {
                            ++binding; // go to next binding slot
                        }

                        src_range_starts.push_back(std::get<0>(*image_view->srv));
                        src_range_sizes.push_back(1);

                        const auto dst_handle {
                            D3D12_CPU_DESCRIPTOR_HANDLE {
                                binding->second.start_cbv_srv_uav.ptr + array_elem * handle_size_cbv_srv_uav
                            }
                        };
                        dst_range_starts.push_back(dst_handle);
                        dst_range_sizes.push_back(1); // TODO: could batch a bit but minor.., else remove these ranges

                        array_elem += 1;
                    }
                } break;
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
                    auto image_views { span<const VkDescriptorImageInfo>(write.pImageInfo, write.descriptorCount) };
                    for (auto i : range(write.descriptorCount)) {
                        auto const& view { image_views[i] };
                        auto image_view { reinterpret_cast<image_view_t *>(view.imageView) };

                        if (array_elem >= binding->second.num_descriptors) {
                            ++binding; // go to next binding slot
                        }

                        src_range_starts.push_back(std::get<0>(*image_view->uav));
                        src_range_sizes.push_back(1);

                        const auto dst_handle {
                            D3D12_CPU_DESCRIPTOR_HANDLE {
                                binding->second.start_cbv_srv_uav.ptr + array_elem * handle_size_cbv_srv_uav
                            }
                        };
                        dst_range_starts.push_back(dst_handle);
                        dst_range_sizes.push_back(1); // TODO: could batch a bit but minor.., else remove these ranges

                        array_elem += 1;
                    }
                } break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
                    auto buffer_views { span<const VkDescriptorBufferInfo>(write.pBufferInfo, write.descriptorCount) };
                    for (auto i : range(write.descriptorCount)) {
                        auto const& view { buffer_views[i] };
                        auto buffer { reinterpret_cast<buffer_t *>(view.buffer) };

                        if (array_elem >= binding->second.num_descriptors) {
                            ++binding; // go to next binding slot
                        }

                        // Uniform buffers are handled via constant buffer views,
                        // requires dynamic creation.
                        auto size {
                            view.range == VK_WHOLE_SIZE ?
                                buffer->memory_requirements.size - view.offset :
                                view.range
                        };

                        // Making the size field of buffer requirements for uniform
                        // buffers a multiple of 256 and setting the required offset
                        // alignment to 256 allows us to patch the size here.
                        // We can always enforce the size to be aligned to 256 for
                        // CBVs without going out-of-bounds.
                        size = (size + 255) & ~255;

                        const D3D12_CONSTANT_BUFFER_VIEW_DESC desc {
                            buffer->resource->GetGPUVirtualAddress() + view.offset,
                            static_cast<UINT>(size)
                        };
                        auto src_handle { device->descriptors_cpu_cbv_srv_uav.alloc() };
                        (*device)->CreateConstantBufferView(&desc, std::get<0>(src_handle));

                        temporary_handles.push_back(src_handle);
                        src_range_starts.push_back(std::get<0>(src_handle));
                        src_range_sizes.push_back(1);

                        const auto dst_handle {
                            D3D12_CPU_DESCRIPTOR_HANDLE {
                                binding->second.start_cbv_srv_uav.ptr + array_elem * handle_size_cbv_srv_uav
                            }
                        };
                        dst_range_starts.push_back(dst_handle);
                        dst_range_sizes.push_back(1); // TODO: could batch a bit but minor.., else remove these ranges

                        array_elem += 1;
                    }
                } break;
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
                    auto buffer_views { span<const VkDescriptorBufferInfo>(write.pBufferInfo, write.descriptorCount) };
                    for (auto i : range(write.descriptorCount)) {
                        auto const& view { buffer_views[i] };
                        auto buffer { reinterpret_cast<buffer_t *>(view.buffer) };

                        if (array_elem >= binding->second.num_descriptors) {
                            ++binding; // go to next binding slot
                        }

                        // Storage buffers are handled via UAV requires dynamic creation.
                        auto size {
                            view.range == VK_WHOLE_SIZE ?
                                buffer->memory_requirements.size - view.offset :
                                view.range
                        };

                        D3D12_UNORDERED_ACCESS_VIEW_DESC desc {
                            DXGI_FORMAT_R32_TYPELESS,
                            D3D12_UAV_DIMENSION_BUFFER,
                        };
                        desc.Buffer = D3D12_BUFFER_UAV {
                            view.offset / 4,
                            static_cast<UINT>(size / 4),
                            0,
                            0,
                            D3D12_BUFFER_UAV_FLAG_RAW,
                        };

                        auto src_handle { device->descriptors_cpu_cbv_srv_uav.alloc() };
                        (*device)->CreateUnorderedAccessView(buffer->resource.Get(), nullptr, &desc, std::get<0>(src_handle));

                        temporary_handles.push_back(src_handle);
                        src_range_starts.push_back(std::get<0>(src_handle));
                        src_range_sizes.push_back(1);

                        const auto dst_handle {
                            D3D12_CPU_DESCRIPTOR_HANDLE {
                                binding->second.start_cbv_srv_uav.ptr + array_elem * handle_size_cbv_srv_uav
                            }
                        };
                        dst_range_starts.push_back(dst_handle);
                        dst_range_sizes.push_back(1); // TODO: could batch a bit but minor.., else remove these ranges

                        array_elem += 1;
                    }
                } break;
                default: {
                    WARN("Unhandled descriptor type: {}", write.descriptorType);
                }
            }
        }
    );
    write_descriptors_set(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        [&] (VkWriteDescriptorSet const& write,
            std::vector<D3D12_CPU_DESCRIPTOR_HANDLE>& dst_range_starts,
            std::vector<D3D12_CPU_DESCRIPTOR_HANDLE>& src_range_starts,
            std::vector<UINT>& dst_range_sizes,
            std::vector<UINT>& src_range_sizes
        ) {
            auto descriptor_set { reinterpret_cast<descriptor_set_t *>(write.dstSet) };
            auto binding { descriptor_set->bindings.find(write.dstBinding) };
            auto array_elem { write.dstArrayElement };

            switch (write.descriptorType) {
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_SAMPLER: {
                    auto image_views { span<const VkDescriptorImageInfo>(write.pImageInfo, write.descriptorCount) };
                    for (auto i : range(write.descriptorCount)) {
                        auto const& view { image_views[i] };
                        auto sampler { reinterpret_cast<sampler_t *>(view.sampler) };

                        if (array_elem >= binding->second.num_descriptors) {
                            ++binding; // go to next binding slot
                        }

                        src_range_starts.push_back(std::get<0>(sampler->sampler));
                        src_range_sizes.push_back(1);

                        const auto dst_handle {
                            D3D12_CPU_DESCRIPTOR_HANDLE {
                                binding->second.start_sampler.ptr + array_elem * handle_size_sampler
                            }
                        };
                        dst_range_starts.push_back(dst_handle);
                        dst_range_sizes.push_back(1); // TODO: could batch a bit but minor.., else remove these ranges

                        array_elem += 1;
                    }
                } break;
            }
        }
    );

    // Copies second
    // TODO
    for (auto i : range(descriptorCopyCount)) {
        auto const& copy { pDescriptorCopies[i] };
    }

    for (auto && view : temporary_handles) {
        auto [handle, index] = view;
        device->descriptors_cpu_cbv_srv_uav.free(handle, index);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(
    VkDevice                                    device,
    const VkFramebufferCreateInfo*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFramebuffer*                              pFramebuffer
) {
    TRACE("vkCreateFramebuffer");
    auto const& info { *pCreateInfo };
    auto attachments { span<const VkImageView>(info.pAttachments, info.attachmentCount) };

    auto framebuffer = new framebuffer_t;
    for (auto const& attachment : attachments) {
        framebuffer->attachments.emplace_back(reinterpret_cast<image_view_t *>(attachment));
    }

    *pFramebuffer = reinterpret_cast<VkFramebuffer>(framebuffer);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyFramebuffer(
    VkDevice                                    device,
    VkFramebuffer                               framebuffer,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroyFramebuffer unimplemented");
    if (framebuffer != VK_NULL_HANDLE) {
        delete reinterpret_cast<framebuffer_t *>(framebuffer);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(
    VkDevice                                    device,
    const VkRenderPassCreateInfo*               pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkRenderPass*                               pRenderPass
) {
    TRACE("vkCreateRenderPass");

    auto const& info { *pCreateInfo };
    auto subpasses { span<const VkSubpassDescription>(info.pSubpasses, info.subpassCount) };

    auto render_pass = new render_pass_t;

    std::vector<std::vector<D3D12_RESOURCE_STATES>> attachment_states;

    for (auto i : range(info.attachmentCount)) {
        render_pass->attachments.emplace_back(
            render_pass_t::attachment_t { info.pAttachments[i], std::nullopt }
        );
        // TODO: resource state
        std::vector<D3D12_RESOURCE_STATES> states(info.subpassCount, D3D12_RESOURCE_STATE_COMMON); // initial layout
        states.push_back(D3D12_RESOURCE_STATE_COMMON); // final layout

        attachment_states.emplace_back(states);
    }

    for (auto sp : range(info.subpassCount)) {
        auto const& subpass { subpasses[sp] };

        render_pass_t::subpass_t pass;
        for (auto i : range(subpass.colorAttachmentCount)) {
            const auto attachment { subpass.pColorAttachments[i] };
            if (attachment.attachment != VK_ATTACHMENT_UNUSED) {
                if (!render_pass->attachments[attachment.attachment].first_use) {
                    render_pass->attachments[attachment.attachment].first_use = sp;
                }
            }
            pass.color_attachments.emplace_back(attachment);
        }
        for (auto i : range(subpass.inputAttachmentCount)) {
            const auto attachment { subpass.pInputAttachments[i] };
            if (attachment.attachment != VK_ATTACHMENT_UNUSED) {
                if (!render_pass->attachments[attachment.attachment].first_use) {
                    render_pass->attachments[attachment.attachment].first_use = sp;
                }
            }
            pass.input_attachments.emplace_back(attachment);
        }

        if (subpass.pDepthStencilAttachment) {
            pass.depth_attachment = *subpass.pDepthStencilAttachment;
            if (!render_pass->attachments[pass.depth_attachment.attachment].first_use) {
                render_pass->attachments[pass.depth_attachment.attachment].first_use = sp;
            }
        } else {
            pass.depth_attachment = VkAttachmentReference { VK_ATTACHMENT_UNUSED };
        }
        // TODO: preserve

        if (subpass.pResolveAttachments) {
            for (auto i : range(subpass.colorAttachmentCount)) {
                const auto attachment { subpass.pResolveAttachments[i] };
                if (attachment.attachment != VK_ATTACHMENT_UNUSED) {
                    if (!render_pass->attachments[attachment.attachment].first_use) {
                        render_pass->attachments[attachment.attachment].first_use = sp;
                    }
                }
                pass.resolve_attachments.emplace_back(attachment);
            }
        }

        render_pass->subpasses.emplace_back(pass);
    }

    *pRenderPass = reinterpret_cast<VkRenderPass>(render_pass);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyRenderPass(
    VkDevice                                    device,
    VkRenderPass                                renderPass,
    const VkAllocationCallbacks*                pAllocator
) {
    WARN("vkDestroyRenderPass unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkGetRenderAreaGranularity(
    VkDevice                                    device,
    VkRenderPass                                renderPass,
    VkExtent2D*                                 pGranularity
) {
    WARN("vkGetRenderAreaGranularity unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(
    VkDevice                                    _device,
    const VkCommandPoolCreateInfo*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkCommandPool*                              pCommandPool
) {
    TRACE("vkCreateCommandPool");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pCreateInfo };

    D3D12_COMMAND_LIST_TYPE list_type;
    switch (info.queueFamilyIndex) {
        case QUEUE_FAMILY_GENERAL_PRESENT:
        case QUEUE_FAMILY_GENERAL:
            list_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            break;
        case QUEUE_FAMILY_COMPUTE:
            list_type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            break;
        case QUEUE_FAMILY_COPY:
            list_type = D3D12_COMMAND_LIST_TYPE_COPY;
            break;
        default:
            ERR("Unexpected queue family index {}", info.queueFamilyIndex);
            assert(true);
    }

    ComPtr<ID3D12CommandAllocator> allocator { nullptr };
    const auto hr {
        (*device)->CreateCommandAllocator(
            list_type,
            IID_PPV_ARGS(&allocator)
        )
    };

    // TODO: errors

    *pCommandPool = reinterpret_cast<VkCommandPool>(new command_pool_t { list_type, allocator });

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    const VkAllocationCallbacks*                pAllocator
) {
    WARN("vkDestroyCommandPool unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolResetFlags                     flags
) {
    WARN("vkResetCommandPool unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(
    VkDevice                                    _device,
    const VkCommandBufferAllocateInfo*          pAllocateInfo,
    VkCommandBuffer*                            pCommandBuffers
) {
    TRACE("vkAllocateCommandBuffers");

    auto device { reinterpret_cast<device_t *>(_device) };
    auto const& info { *pAllocateInfo };
    auto const& pool { *reinterpret_cast<command_pool_t *>(info.commandPool) };

    for (auto i : range(info.commandBufferCount)) {
        auto command_buffer {
            new command_buffer_t(
                device->descriptors_gpu_cbv_srv_uav.heap(),
                device->descriptors_gpu_sampler.heap()
            )
        };

        ID3D12GraphicsCommandList2* command_list { nullptr };
        {
            const auto hr {
                (*device)->CreateCommandList(
                    0,
                    pool.type,
                    pool.allocator.Get(),
                    nullptr,
                    IID_PPV_ARGS(&command_list)
                )
            };
            // TODO: error handling
        }
        command_list->Close();

        command_buffer->allocator = pool.allocator.Get();
        command_buffer->command_list = command_list;
        command_buffer->_device = device;

        pCommandBuffers[i] = reinterpret_cast<VkCommandBuffer>(command_buffer);
    }

    // TODO:

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    uint32_t                                    commandBufferCount,
    const VkCommandBuffer*                      pCommandBuffers
) {
    TRACE("vkFreeCommandBuffers");
    WARN("vkFreeCommandBuffers unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(
    VkCommandBuffer                             _commandBuffer,
    const VkCommandBufferBeginInfo*             pBeginInfo
) {
    TRACE("vkBeginCommandBuffer");

    // TODO: begin info

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    command_buffer->reset();

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(
    VkCommandBuffer                             _commandBuffer
) {
    TRACE("vkEndCommandBuffer");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    command_buffer->end();

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(
    VkCommandBuffer                             commandBuffer,
    VkCommandBufferResetFlags                   flags
) {
    TRACE("vkResetCommandBuffer");
    WARN("vkResetCommandBuffer unimplemented");
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(
    VkCommandBuffer                             _commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipeline                                  _pipeline
) {
    TRACE("vkCmdBindPipeline");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto pipeline { reinterpret_cast<pipeline_t *>(_pipeline) };
    auto signature { pipeline->signature };

    switch (pipelineBindPoint) {
        case VK_PIPELINE_BIND_POINT_GRAPHICS: {
            if (!command_buffer->active_slot) {
                std::array<ID3D12DescriptorHeap *const, 2> heaps {
                    command_buffer->heap_cbv_srv_uav,
                    command_buffer->heap_sampler
                };
                (*command_buffer)->SetDescriptorHeaps(2, &heaps[0]);
                command_buffer->active_slot = command_buffer_t::SLOT_GRAPHICS;
            }

            if (command_buffer->graphics_slot.signature != signature) {
                (*command_buffer)->SetGraphicsRootSignature(signature);
                command_buffer->graphics_slot.signature = signature;
                // TODO: descriptor sets
            }
            command_buffer->graphics_slot.pipeline = pipeline;
            (*command_buffer)->IASetPrimitiveTopology(pipeline->topology); // no need to cache this

            for (auto i : range(MAX_VERTEX_BUFFER_SLOTS)) {
                command_buffer->vertex_buffer_views[i].StrideInBytes = pipeline->vertex_strides[i];
            }

            // Apply static states
            if (pipeline->static_viewports) {
                (*command_buffer)->RSSetViewports(
                    static_cast<UINT>(pipeline->static_viewports->size()),
                    pipeline->static_viewports->data()
                );
                command_buffer->viewports_dirty = false;
            }
            if (pipeline->static_scissors) {
                (*command_buffer)->RSSetScissorRects(
                    static_cast<UINT>(pipeline->static_scissors->size()),
                    pipeline->static_scissors->data()
                );
                command_buffer->scissors_dirty = false;
            }
            if (pipeline->static_blend_factors) {
                (*command_buffer)->OMSetBlendFactor(
                    pipeline->static_blend_factors->factors
                );
            }
            if (pipeline->static_depth_bounds) {
                const auto [min, max] = *pipeline->static_depth_bounds;
                (*command_buffer)->OMSetDepthBounds(min, max);
            }
            if (pipeline->static_stencil_reference) {
                (*command_buffer)->OMSetStencilRef(
                    *pipeline->static_stencil_reference
                );
            }
            if (!(pipeline->dynamic_states & DYNAMIC_STATE_DEPTH_BIAS)) {
                command_buffer->dynamic_state.depth_bias = pipeline->static_state.depth_bias;
                command_buffer->dynamic_state.depth_bias_clamp = pipeline->static_state.depth_bias_clamp;
                command_buffer->dynamic_state.depth_bias_slope = pipeline->static_state.depth_bias_slope;
            }
            if (!(pipeline->dynamic_states & DYNAMIC_STATE_STENCIL_COMPARE_MASK)) {
                command_buffer->dynamic_state.stencil_read_mask = pipeline->static_state.stencil_read_mask;
            }
            if (!(pipeline->dynamic_states & DYNAMIC_STATE_STENCIL_WRITE_MASK)) {
                command_buffer->dynamic_state.stencil_write_mask = pipeline->static_state.stencil_write_mask;
            }
        } break;
        case VK_PIPELINE_BIND_POINT_COMPUTE: {
            if (!command_buffer->active_slot) {
                std::array<ID3D12DescriptorHeap *const, 2> heaps {
                    command_buffer->heap_cbv_srv_uav,
                    command_buffer->heap_sampler
                };
                (*command_buffer)->SetDescriptorHeaps(2, &heaps[0]);
                command_buffer->active_slot = command_buffer_t::SLOT_COMPUTE;
            }

            if (command_buffer->compute_slot.signature != signature) {
                (*command_buffer)->SetComputeRootSignature(signature);
                command_buffer->compute_slot.signature = signature;
                // TODO: descriptor sets
            }
            command_buffer->compute_slot.pipeline = pipeline;
        } break;
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetViewport(
    VkCommandBuffer                             _commandBuffer,
    uint32_t                                    firstViewport,
    uint32_t                                    viewportCount,
    const VkViewport*                           pViewports
) {
    TRACE("vkCmdSetViewport");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };

    command_buffer->viewports_dirty = true;
    command_buffer->num_viewports_scissors = std::max(
        command_buffer->num_viewports_scissors,
        firstViewport+viewportCount
    );

    // Cache viewports internally and set them on draw
    for (auto i : range(firstViewport, firstViewport+viewportCount)) {
        auto const& vp = pViewports[i];
        command_buffer->viewports[i] = { vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth };
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(
    VkCommandBuffer                             _commandBuffer,
    uint32_t                                    firstScissor,
    uint32_t                                    scissorCount,
    const VkRect2D*                             pScissors
) {
    TRACE("vkCmdSetScissor");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };

    command_buffer->scissors_dirty = true;
    command_buffer->num_viewports_scissors = std::max(
        command_buffer->num_viewports_scissors,
        firstScissor+scissorCount
    );

    // Cache viewports internally and set them on draw
    for (auto i : range(firstScissor, firstScissor+scissorCount)) {
        auto const& sc = pScissors[i];
        command_buffer->scissors[i] = {
            sc.offset.x,
            sc.offset.y,
            static_cast<LONG>(sc.offset.x + sc.extent.width),
            static_cast<LONG>(sc.offset.y + sc.extent.height)
        };
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetLineWidth(
    VkCommandBuffer                             commandBuffer,
    float                                       lineWidth
) {
    TRACE("vkCmdSetLineWidth");

    // Nothing to do, not supported
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBias(
    VkCommandBuffer                             _commandBuffer,
    float                                       depthBiasConstantFactor,
    float                                       depthBiasClamp,
    float                                       depthBiasSlopeFactor
) {
    TRACE("vkCmdSetDepthBias");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    command_buffer->dynamic_state.depth_bias = static_cast<INT>(depthBiasConstantFactor);
    command_buffer->dynamic_state.depth_bias_clamp = depthBiasClamp;
    command_buffer->dynamic_state.depth_bias_slope = depthBiasSlopeFactor;
    command_buffer->dynamic_state_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetBlendConstants(
    VkCommandBuffer                             _commandBuffer,
    const float                                 blendConstants[4]
) {
    TRACE("vkCmdSetBlendConstants");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    (*command_buffer)->OMSetBlendFactor(blendConstants);
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBounds(
    VkCommandBuffer                             _commandBuffer,
    float                                       minDepthBounds,
    float                                       maxDepthBounds
) {
    TRACE("vkCmdSetDepthBounds");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    (*command_buffer)->OMSetDepthBounds(minDepthBounds, maxDepthBounds);
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilCompareMask(
    VkCommandBuffer                             _commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    compareMask
) {
    TRACE("vkCmdSetStencilCompareMask");

    // PORTABILITY: compareMask must be same for both faces
    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    command_buffer->dynamic_state.stencil_read_mask = compareMask;
    command_buffer->dynamic_state_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilWriteMask(
    VkCommandBuffer                             _commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    writeMask
) {
    TRACE("vkCmdSetStencilWriteMask");

    // PORTABILITY: compareMask must be same for both faces
    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    command_buffer->dynamic_state.stencil_write_mask = writeMask;
    command_buffer->dynamic_state_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilReference(
    VkCommandBuffer                             _commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    reference
) {
    TRACE("vkCmdSetStencilReference");

    // PORTABILITY: compareMask must be same for both faces
    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    (*command_buffer)->OMSetStencilRef(reference);
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(
    VkCommandBuffer                             _commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            _layout,
    uint32_t                                    firstSet,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet*                      pDescriptorSets,
    uint32_t                                    dynamicOffsetCount,
    const uint32_t*                             pDynamicOffsets
) {
    TRACE("vkCmdBindDescriptorSets");

    // TODO: dynamic

    auto descriptor_sets { span<const VkDescriptorSet>(pDescriptorSets, descriptorSetCount) };
    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto layout { reinterpret_cast<pipeline_layout_t *>(_layout) };

    const auto start_cbv_srv_uav {
        command_buffer->heap_cbv_srv_uav->GetGPUDescriptorHandleForHeapStart()
    };
    const auto start_sampler {
        command_buffer->heap_sampler->GetGPUDescriptorHandleForHeapStart()
    };

    auto bind_descriptor_set = [&] (command_buffer_t::pipeline_slot_t& pipeline) {
        // Find the table entry corresponding with `first_set`
        auto entry { layout->num_root_constants };
        for (auto i : range(firstSet)) {
            const auto entry_type { layout->tables[i] };
            if (entry_type & TABLE_CBV_SRV_UAV) {
                entry += 1;
            }
            if (entry_type & TABLE_SAMPLER) {
                entry += 1;
            }
        }

        for (auto i : range(descriptorSetCount)) {
            auto set { reinterpret_cast<descriptor_set_t *>(descriptor_sets[i]) };
            auto const& table { layout->tables[firstSet+i] };

            if (set->start_cbv_srv_uav) {
                // Only storing relative address, less storage needed
                const auto set_offset { set->start_cbv_srv_uav->ptr - start_cbv_srv_uav.ptr };
                pipeline.root_data.set_cbv_srv_uav(entry, static_cast<uint32_t>(set_offset));

                entry += 1;
            }
            if (set->start_sampler) {
                // Only storing relative address, less storage needed
                const auto set_offset { set->start_sampler->ptr - start_sampler.ptr };
                pipeline.root_data.set_sampler(entry, static_cast<uint32_t>(set_offset));

                entry += 1;
            }
        }
    };

    switch (pipelineBindPoint) {
        case VK_PIPELINE_BIND_POINT_GRAPHICS: {
            bind_descriptor_set(command_buffer->graphics_slot);
        } break;
        case VK_PIPELINE_BIND_POINT_COMPUTE: {
            bind_descriptor_set(command_buffer->compute_slot);
        } break;
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(
    VkCommandBuffer                             _commandBuffer,
    VkBuffer                                    _buffer,
    VkDeviceSize                                offset,
    VkIndexType                                 indexType
) {
    TRACE("vkCmdBindIndexBuffer");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto buffer { reinterpret_cast<buffer_t *>(_buffer) };

    DXGI_FORMAT index_format { DXGI_FORMAT_UNKNOWN };
    switch (indexType) {
        case VK_INDEX_TYPE_UINT16: index_format = DXGI_FORMAT_R16_UINT; break;
        case VK_INDEX_TYPE_UINT32: index_format = DXGI_FORMAT_R32_UINT; break;
    }

    (*command_buffer)->IASetIndexBuffer(
        &D3D12_INDEX_BUFFER_VIEW {
            buffer->resource->GetGPUVirtualAddress() + offset,
            static_cast<UINT>(buffer->memory_requirements.size - offset),
            index_format
        }
    );

    command_buffer->index_type = indexType;
    command_buffer->dynamic_state_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers(
    VkCommandBuffer                             _commandBuffer,
    uint32_t                                    firstBinding,
    uint32_t                                    bindingCount,
    const VkBuffer*                             pBuffers,
    const VkDeviceSize*                         pOffsets
) {
    TRACE("vkCmdBindVertexBuffers");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };

    for (auto i : range(firstBinding, firstBinding + bindingCount)) {
        auto buffer { reinterpret_cast<buffer_t *>(pBuffers[i-firstBinding]) };
        auto offset { pOffsets[i-firstBinding] };

        const auto address { buffer->resource->GetGPUVirtualAddress() };

        command_buffer->vertex_buffer_views[i].BufferLocation = address + offset;
        command_buffer->vertex_buffer_views[i].SizeInBytes = buffer->memory_requirements.size - offset;
        // stride information inside the graphics pipeline

        command_buffer->vertex_buffer_views_dirty.set(i);
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdDraw(
    VkCommandBuffer                             _commandBuffer,
    uint32_t                                    vertexCount,
    uint32_t                                    instanceCount,
    uint32_t                                    firstVertex,
    uint32_t                                    firstInstance
) {
    TRACE("vkCmdDraw");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    command_buffer->bind_graphics_slot(draw_type::DRAW);
    (*command_buffer)->DrawInstanced(
        vertexCount,
        instanceCount,
        firstVertex,
        firstInstance
    );
}

VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(
    VkCommandBuffer                             _commandBuffer,
    uint32_t                                    indexCount,
    uint32_t                                    instanceCount,
    uint32_t                                    firstIndex,
    int32_t                                     vertexOffset,
    uint32_t                                    firstInstance
) {
    TRACE("vkCmdDrawIndexed");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    command_buffer->bind_graphics_slot(draw_type::DRAW_INDEXED);
    (*command_buffer)->DrawIndexedInstanced(
        indexCount,
        instanceCount,
        firstIndex,
        vertexOffset,
        firstInstance
    );
}

VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirect(
    VkCommandBuffer                             _commandBuffer,
    VkBuffer                                    _buffer,
    VkDeviceSize                                offset,
    uint32_t                                    drawCount,
    uint32_t                                    stride
) {
    TRACE("vkCmdDrawIndirect");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto buffer { reinterpret_cast<buffer_t *>(_buffer) };

    auto get_signature = [&] () {
        {
            std::shared_lock<std::shared_mutex> lock(command_buffer->_device->draw_indirect_access);
            auto signature { command_buffer->_device->draw_indirect.find(stride) };
            if (signature != command_buffer->_device->draw_indirect.end()) {
                return signature->second.Get();
            }
        }

        auto signature = create_command_signature(
            command_buffer->device(),
            D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
            stride
        );

        std::scoped_lock<std::shared_mutex> lock(command_buffer->_device->draw_indirect_access);
        command_buffer->_device->draw_indirect.emplace(stride, signature);
        return signature.Get();
    };

    command_buffer->bind_graphics_slot(draw_type::DRAW);
    const auto signature = get_signature();
    (*command_buffer)->ExecuteIndirect(
        signature,
        drawCount,
        buffer->resource.Get(),
        offset,
        nullptr,
        0
    );
}

VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(
    VkCommandBuffer                             _commandBuffer,
    VkBuffer                                    _buffer,
    VkDeviceSize                                offset,
    uint32_t                                    drawCount,
    uint32_t                                    stride
) {
    TRACE("vkCmdDrawIndexedIndirect");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto buffer { reinterpret_cast<buffer_t *>(_buffer) };

    auto get_signature = [&] () {
        {
            std::shared_lock<std::shared_mutex> lock(command_buffer->_device->draw_indirect_access);
            auto signature { command_buffer->_device->draw_indexed_indirect.find(stride) };
            if (signature != command_buffer->_device->draw_indexed_indirect.end()) {
                return signature->second.Get();
            }
        }

        auto signature = create_command_signature(
            command_buffer->device(),
            D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
            stride
        );

        std::scoped_lock<std::shared_mutex> lock(command_buffer->_device->draw_indirect_access);
        command_buffer->_device->draw_indexed_indirect.emplace(stride, signature);
        return signature.Get();
    };

    command_buffer->bind_graphics_slot(draw_type::DRAW_INDEXED);
    const auto signature = get_signature();
    (*command_buffer)->ExecuteIndirect(
        signature,
        drawCount,
        buffer->resource.Get(),
        offset,
        nullptr,
        0
    );
}

VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(
    VkCommandBuffer                             _commandBuffer,
    uint32_t                                    groupCountX,
    uint32_t                                    groupCountY,
    uint32_t                                    groupCountZ
) {
    TRACE("vkCmdDispatch");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    command_buffer->bind_compute_slot();
    (*command_buffer)->Dispatch(
        groupCountX,
        groupCountY,
        groupCountZ
    );
}

VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset
) {
    TRACE("vkCmdDispatchIndirect");
    WARN("vkCmdDispatchIndirect unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(
    VkCommandBuffer                             _commandBuffer,
    VkBuffer                                    _srcBuffer,
    VkBuffer                                    _dstBuffer,
    uint32_t                                    regionCount,
    const VkBufferCopy*                         pRegions
) {
    TRACE("vkCmdCopyBuffer");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto src_buffer { reinterpret_cast<buffer_t *>(_srcBuffer) };
    auto dst_buffer { reinterpret_cast<buffer_t *>(_dstBuffer) };
    auto regions { span<const VkBufferCopy>(pRegions, regionCount) };

    for (auto const& region : regions) {
        (*command_buffer)->CopyBufferRegion(
            dst_buffer->resource.Get(),
            region.dstOffset,
            src_buffer->resource.Get(),
            region.srcOffset,
            region.size
        );
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(
    VkCommandBuffer                             _commandBuffer,
    VkImage                                     _srcImage,
    VkImageLayout                               srcImageLayout,
    VkImage                                     _dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkImageCopy*                          pRegions
) {
    TRACE("vkCmdCopyImage");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto src_image { reinterpret_cast<image_t *>(_srcImage) };
    auto dst_image { reinterpret_cast<image_t *>(_dstImage) };
    auto regions { span<const VkImageCopy>(pRegions, regionCount) };

    for (auto const& region : regions) {
        const auto num_layers { region.dstSubresource.layerCount };

        for (auto layer : range(num_layers)) {
            const D3D12_TEXTURE_COPY_LOCATION dst {
                CD3DX12_TEXTURE_COPY_LOCATION(
                    dst_image->resource.Get(),
                    D3D12CalcSubresource(
                        region.dstSubresource.mipLevel,
                        region.dstSubresource.baseArrayLayer + layer,
                        0,
                        dst_image->resource_desc.MipLevels,
                        dst_image->resource_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ?
                            1 :
                            dst_image->resource_desc.DepthOrArraySize
                    )
                )
            };
            const D3D12_TEXTURE_COPY_LOCATION src {
                CD3DX12_TEXTURE_COPY_LOCATION(
                    src_image->resource.Get(),
                    D3D12CalcSubresource(
                        region.srcSubresource.mipLevel,
                        region.srcSubresource.baseArrayLayer + layer,
                        0,
                        src_image->resource_desc.MipLevels,
                        src_image->resource_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ?
                            1 :
                            src_image->resource_desc.DepthOrArraySize
                    )
                )
            };
            const D3D12_BOX box {
                static_cast<UINT>(region.srcOffset.x),
                static_cast<UINT>(region.srcOffset.y),
                static_cast<UINT>(region.srcOffset.z),
                region.srcOffset.x + region.extent.width,
                region.srcOffset.y + region.extent.height,
                region.srcOffset.z + region.extent.depth
            };

            (*command_buffer)->CopyTextureRegion(
                &dst,
                region.dstOffset.x,
                region.dstOffset.y,
                region.dstOffset.z,
                &src,
                &box
            );
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(
    VkCommandBuffer                             _commandBuffer,
    VkImage                                     _srcImage,
    VkImageLayout                               srcImageLayout,
    VkImage                                     _dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkImageBlit*                          pRegions,
    VkFilter                                    filter
) {
    TRACE("vkCmdBlitImage");
    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto src_image { reinterpret_cast<image_t *>(_srcImage) };
    auto dst_image { reinterpret_cast<image_t *>(_dstImage) };
    auto regions { span<const VkImageBlit>(pRegions, regionCount) };

    const auto src_img_width { src_image->resource_desc.Width };
    const auto src_img_height { src_image->resource_desc.Height };
    const auto src_img_depth {
        src_image->resource_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D ?
            1 :
            src_image->resource_desc.DepthOrArraySize
    };

    // TODO: filter

    command_buffer->active_slot = std::nullopt;

    ComPtr<ID3D12DescriptorHeap> temp_heap { nullptr };
    const UINT handle_size {
        command_buffer->device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    };
    {
        const D3D12_DESCRIPTOR_HEAP_DESC desc {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            static_cast<UINT>(2 * regionCount),
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            0u
        };
        auto const hr { command_buffer->device()->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&temp_heap)) };
    }

    command_buffer->temp_heaps.push_back(temp_heap);

    const auto start_cpu = temp_heap->GetCPUDescriptorHandleForHeapStart();
    const auto start_gpu = temp_heap->GetGPUDescriptorHandleForHeapStart();

    const D3D12_CPU_DESCRIPTOR_HANDLE srv_start { start_cpu.ptr };
    const D3D12_CPU_DESCRIPTOR_HANDLE uav_start { start_cpu.ptr + handle_size };

    (*command_buffer)->SetPipelineState(command_buffer->_device->pso_blit_2d.Get());
    (*command_buffer)->SetComputeRootSignature(command_buffer->_device->signature_blit_2d.Get());
    std::array<ID3D12DescriptorHeap *const, 1> heaps { temp_heap.Get() };
    (*command_buffer)->SetDescriptorHeaps(1, &heaps[0]);

    for (auto i : range(regionCount)) {
        auto const& region { regions[i] };

        const D3D12_CPU_DESCRIPTOR_HANDLE srv { srv_start.ptr + 2 * i * handle_size };
        const D3D12_CPU_DESCRIPTOR_HANDLE uav { uav_start.ptr + 2 * i * handle_size };

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc { src_image->resource_desc.Format };
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv_desc.Texture2DArray = D3D12_TEX2D_ARRAY_SRV {
            region.srcSubresource.mipLevel,
            1,
            region.srcSubresource.baseArrayLayer,
            region.srcSubresource.layerCount,
            0,
            0.0,
        };
        srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
            D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0,
            D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1,
            D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2,
            D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3
        );

        command_buffer->device()->CreateShaderResourceView(
            src_image->resource.Get(),
            &srv_desc,
            srv
        );

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc { dst_image->resource_desc.Format };
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uav_desc.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
            region.dstSubresource.mipLevel,
            region.dstSubresource.baseArrayLayer,
            region.srcSubresource.layerCount,
            0,
        };

        command_buffer->device()->CreateUnorderedAccessView(
            dst_image->resource.Get(),
            nullptr, // counter
            &uav_desc,
            uav
        );


        (*command_buffer)->SetComputeRootDescriptorTable(
            0,
            D3D12_GPU_DESCRIPTOR_HANDLE { start_gpu.ptr + 2 * i * handle_size }
        );

        const auto level { region.srcSubresource.mipLevel };
        const auto width { std::max(static_cast<UINT>(src_img_width >> level), static_cast<UINT>(1u)) };
        const auto height { std::max(static_cast<UINT>(src_img_height >> level), static_cast<UINT>(1u)) };
        const auto depth { std::max(static_cast<UINT>(src_img_depth >> level), static_cast<UINT>(1u)) };

        std::array<uint32_t, 15> constant_data {
            region.srcOffsets[0].x, region.srcOffsets[0].y, region.srcOffsets[0].z,
            region.srcOffsets[1].x - region.srcOffsets[0].x,
            region.srcOffsets[1].y - region.srcOffsets[0].y,
            region.srcOffsets[1].z - region.srcOffsets[0].z,
            region.dstOffsets[0].x, region.dstOffsets[0].y, region.dstOffsets[0].z,
            region.dstOffsets[1].x - region.dstOffsets[0].x,
            region.dstOffsets[1].y - region.dstOffsets[0].y,
            region.dstOffsets[1].z - region.dstOffsets[0].z,
            width,
            height,
            depth,
        };
        (*command_buffer)->SetComputeRoot32BitConstants(
            1,
            static_cast<UINT>(constant_data.size()),
            constant_data.data(),
            0
        );

        (*command_buffer)->Dispatch(
            region.dstOffsets[1].x - region.dstOffsets[0].x,
            region.dstOffsets[1].y - region.dstOffsets[0].y,
            region.dstOffsets[1].z - region.dstOffsets[0].z
        );

        std::array<D3D12_RESOURCE_BARRIER, 2> uav_barriers {
            CD3DX12_RESOURCE_BARRIER::UAV(src_image->resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(dst_image->resource.Get())
        };
        (*command_buffer)->ResourceBarrier(uav_barriers.size(), uav_barriers.data());
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(
    VkCommandBuffer                             _commandBuffer,
    VkBuffer                                    _srcBuffer,
    VkImage                                     _dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkBufferImageCopy*                    pRegions
) {
    TRACE("vkCmdCopyBufferToImage");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto src_buffer { reinterpret_cast<buffer_t *>(_srcBuffer) };
    auto dst_image { reinterpret_cast<image_t *>(_dstImage) };
    auto regions { span<const VkBufferImageCopy>(pRegions, regionCount) };

    const auto img_width { dst_image->resource_desc.Width };
    const auto img_height { dst_image->resource_desc.Height };
    const auto img_depth {
        dst_image->resource_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D ?
            1 :
            dst_image->resource_desc.DepthOrArraySize
    };

    // In-case we have unaligned pitches
    struct cs_copy_region {
        UINT offset;
        UINT row_pitch;
        UINT slice_pitch;
        UINT mip_level;
        UINT array_layer;
        UINT bits_format;
        VkExtent3D extent;
    };
    std::vector<cs_copy_region> cs_regions {};

    for (auto const& region : regions) {
        const auto level { region.imageSubresource.mipLevel };
        const auto width { std::max(static_cast<UINT>(img_width >> level), static_cast<UINT>(1u)) };
        const auto height { std::max(static_cast<UINT>(img_height >> level), static_cast<UINT>(1u)) };
        const auto depth { std::max(static_cast<UINT>(img_depth >> level), static_cast<UINT>(1u)) };

        const auto base_layer { region.imageSubresource.baseArrayLayer };
        const auto num_layers { region.imageSubresource.layerCount };

        for (auto layer : range(base_layer, base_layer+num_layers)) {
            const auto buffer_width {
                region.bufferRowLength ? region.bufferRowLength : region.imageExtent.width
            };
            const auto buffer_height {
                region.bufferImageHeight ? region.bufferImageHeight : region.imageExtent.height
            };

            // Aligning, in particular required for the case of block compressed formats and non-multiple width/height fields.
            // TODO: verify the align parts, not totally confident it's correct in all circumstances
            const auto byte_per_texel { dst_image->block_data.bits / 8 };
            const auto num_rows { up_align(buffer_height, dst_image->block_data.height) / dst_image->block_data.height };
            const auto raw_row_pitch {
                (up_align(buffer_width, dst_image->block_data.width) / dst_image->block_data.width) * byte_per_texel
            };
            const auto raw_slice_pitch { raw_row_pitch * num_rows };
            const auto offset { region.bufferOffset + (layer - base_layer) * raw_slice_pitch * depth };

            const auto pitch_aligned {
                (raw_row_pitch & (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT-1)) == 0
            };
            const auto offset_aligned {
                (offset & (D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT-1)) == 0
            };

            struct copy_region_t {
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT src_desc;
                UINT dst_x;
                UINT dst_y;
                UINT dst_z;
                D3D12_BOX box;
            };

            std::vector<copy_region_t> new_regions;

            if ((pitch_aligned || num_rows == 1)) {
                // Interesting for the case of a single row, where we can just increase the pitch to whatever suites us
                const auto single_row_pitch { up_align(raw_row_pitch, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) };
                const auto row_pitch { num_rows > 1 ? raw_row_pitch : single_row_pitch };
                const auto slice_pitch { row_pitch * num_rows };

                if (offset_aligned) {
                    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT src_desc {
                        offset,
                        D3D12_SUBRESOURCE_FOOTPRINT {
                            dst_image->resource_desc.Format,
                            up_align(width, dst_image->block_data.width),
                            up_align(height, dst_image->block_data.height),
                            depth,
                            row_pitch,
                        }
                    };

                    new_regions.emplace_back(
                        copy_region_t {
                            src_desc,
                            static_cast<UINT>(region.imageOffset.x),
                            static_cast<UINT>(region.imageOffset.y),
                            static_cast<UINT>(region.imageOffset.z),
                            D3D12_BOX {
                                0, 0, 0,
                                up_align(region.imageExtent.width, dst_image->block_data.width),
                                up_align(region.imageExtent.height, dst_image->block_data.height),
                                region.imageExtent.depth
                            },
                        }
                    );
                } else {
                    const auto aligned_offset {
                        offset & ~(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1)
                    };

                    // Texture splitting algorithm from NXT, adopted to coding style
                    // LICENSE in the header
                    auto calc_texel_offsets = [] (
                        uint32_t offset, uint32_t row_pitch, uint32_t slice_pitch, uint32_t texel_size,
                        uint32_t& texel_offset_x, uint32_t& texel_offset_y, uint32_t& texel_offset_z,
                        uint32_t block_width, uint32_t block_height
                    ) {
                        const uint32_t byte_offset_x = offset % row_pitch;
                        offset -= byte_offset_x;
                        const uint32_t byte_offset_y = offset % slice_pitch;
                        const uint32_t byte_offset_z = offset - byte_offset_y;

                        texel_offset_x = (byte_offset_x / texel_size) * block_width;
                        texel_offset_y = (byte_offset_y / row_pitch ) * block_height;
                        texel_offset_z = byte_offset_z / slice_pitch;
                    };

                    uint32_t texel_offset_x, texel_offset_y, texel_offset_z;
                    calc_texel_offsets(
                        static_cast<UINT>(offset - aligned_offset),
                        row_pitch,
                        slice_pitch,
                        byte_per_texel,
                        texel_offset_x,
                        texel_offset_y,
                        texel_offset_z,
                        dst_image->block_data.width,
                        dst_image->block_data.height
                    );
                    uint32_t row_pitch_texels = dst_image->block_data.width * row_pitch / byte_per_texel;

                    if (region.imageExtent.width + texel_offset_x <= row_pitch_texels) {
                        const D3D12_PLACED_SUBRESOURCE_FOOTPRINT src_desc {
                            aligned_offset,
                            D3D12_SUBRESOURCE_FOOTPRINT {
                                dst_image->resource_desc.Format,
                                texel_offset_x + up_align(width, dst_image->block_data.width),
                                texel_offset_y + up_align(height, dst_image->block_data.height),
                                texel_offset_z + depth,
                                row_pitch,
                            }
                        };

                        new_regions.emplace_back(
                            copy_region_t {
                                src_desc,
                                static_cast<UINT>(region.imageOffset.x),
                                static_cast<UINT>(region.imageOffset.y),
                                static_cast<UINT>(region.imageOffset.z),
                                D3D12_BOX {
                                    texel_offset_x,
                                    texel_offset_y,
                                    texel_offset_z,
                                    texel_offset_x + up_align(region.imageExtent.width, dst_image->block_data.width),
                                    texel_offset_y + up_align(region.imageExtent.height, dst_image->block_data.height),
                                    texel_offset_z + region.imageExtent.depth,
                                },
                            }
                        );
                    } else {
                        {
                            const D3D12_PLACED_SUBRESOURCE_FOOTPRINT src_desc {
                                aligned_offset,
                                D3D12_SUBRESOURCE_FOOTPRINT {
                                    dst_image->resource_desc.Format,
                                    row_pitch_texels,
                                    texel_offset_y + up_align(height, dst_image->block_data.height),
                                    texel_offset_z + depth,
                                    row_pitch,
                                }
                            };

                            new_regions.emplace_back(
                                copy_region_t {
                                    src_desc,
                                    static_cast<UINT>(region.imageOffset.x),
                                    static_cast<UINT>(region.imageOffset.y),
                                    static_cast<UINT>(region.imageOffset.z),
                                    D3D12_BOX {
                                        texel_offset_x,
                                        texel_offset_y,
                                        texel_offset_z,
                                        row_pitch_texels,
                                        texel_offset_y + up_align(region.imageExtent.height, dst_image->block_data.height),
                                        texel_offset_z + region.imageExtent.depth,
                                    },
                                }
                            );
                        }
                        {
                            const D3D12_PLACED_SUBRESOURCE_FOOTPRINT src_desc {
                                aligned_offset,
                                D3D12_SUBRESOURCE_FOOTPRINT {
                                    dst_image->resource_desc.Format,
                                    up_align(width, dst_image->block_data.width) - row_pitch_texels + texel_offset_x,
                                    up_align(height, dst_image->block_data.height) + texel_offset_y + dst_image->block_data.height,
                                    depth + texel_offset_z,
                                    row_pitch,
                                }
                            };

                            new_regions.emplace_back(
                                copy_region_t {
                                    src_desc,
                                    static_cast<UINT>(region.imageOffset.x + row_pitch_texels - texel_offset_x),
                                    static_cast<UINT>(region.imageOffset.y),
                                    static_cast<UINT>(region.imageOffset.z),
                                    D3D12_BOX {
                                        0,
                                        texel_offset_y + dst_image->block_data.height,
                                        texel_offset_z,
                                        up_align(width, dst_image->block_data.width) - row_pitch_texels + texel_offset_x,
                                        region.imageExtent.height + texel_offset_y + dst_image->block_data.height,
                                        region.imageExtent.depth + texel_offset_z,
                                    },
                                }
                            );
                        }
                    }
                    // End texture splitting
                }
            } else if (false) { // TODO: check if the format supports UAV
                // yay! compute shaders.... why???
                // We will *never* hit this part on a copy queue except someone violates our
                // exposed image granularity requirements

                cs_regions.emplace_back(
                    cs_copy_region {
                        static_cast<UINT>(offset),
                        raw_row_pitch,
                        raw_slice_pitch,
                        level,
                        layer,
                        dst_image->block_data.bits,
                        region.imageExtent,
                    }
                );
            } else {
                // Ultra-slow path, where we can't even use compute shaders..
                // Manually stitching the texture together with one copy per buffer line
                // TODO: 3d textures heh
                const auto aligned_offset {
                    static_cast<UINT>(offset) & ~(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1u)
                };
                const auto row_pitch { up_align(raw_row_pitch, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) };
                const auto aligned_row_pitch { up_align(row_pitch, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT) };
                const auto buffer_width { dst_image->block_data.width * row_pitch / byte_per_texel };
                auto buffer_height { height * raw_row_pitch / row_pitch };

                const auto diff_offset { static_cast<UINT>(offset) - aligned_offset };
                if (diff_offset > 0) {
                    buffer_height += 2; // Up to two additional rows due to ALIGNMENT offset? // TODO
                }

                for (auto y : range(num_rows)) {
                    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT src_desc {
                        aligned_offset,
                        D3D12_SUBRESOURCE_FOOTPRINT {
                            dst_image->resource_desc.Format,
                            buffer_width,
                            up_align(std::max(1u, buffer_height), dst_image->block_data.height),
                            1,
                            row_pitch,
                        }
                    };

                    const auto real_offset { diff_offset + y * raw_row_pitch };
                    const auto aligned_real_offset { real_offset & ~(row_pitch - 1) };
                    const auto offset_x { real_offset - aligned_real_offset };
                    const auto offset_y { ((y * raw_row_pitch) & ~(row_pitch - 1)) / row_pitch };

                    const auto buffer_texels { offset_x / byte_per_texel * dst_image->block_data.width };
                    const auto default_region_width { dst_image->block_data.width * raw_row_pitch / byte_per_texel };
                    const auto region_width = std::min(buffer_width, buffer_texels + default_region_width);

                    new_regions.emplace_back(
                        copy_region_t {
                            src_desc,
                            static_cast<UINT>(region.imageOffset.x),
                            static_cast<UINT>(region.imageOffset.y) + y * dst_image->block_data.height,
                            static_cast<UINT>(region.imageOffset.z),
                            D3D12_BOX {
                                buffer_texels,
                                offset_y * dst_image->block_data.height,
                                0,
                                up_align(region_width, dst_image->block_data.width),
                                (offset_y + 1) * dst_image->block_data.height,
                                1,
                            },
                        }
                    );

                    if (buffer_width < buffer_texels + default_region_width) {
                        // Splitted region, need to stitch again from the next line
                        new_regions.emplace_back(
                            copy_region_t {
                                src_desc,
                                static_cast<UINT>(region.imageOffset.x) + buffer_width - buffer_texels,
                                static_cast<UINT>(region.imageOffset.y) + y * dst_image->block_data.height,
                                static_cast<UINT>(region.imageOffset.z),
                                D3D12_BOX {
                                    0,
                                    (offset_y + 1) * dst_image->block_data.height,
                                    0,
                                    (buffer_texels + default_region_width - buffer_width),
                                    (offset_y + 2) * dst_image->block_data.height,
                                    1,
                                },
                            }
                        );
                    }
                }

            };

            const D3D12_TEXTURE_COPY_LOCATION dst_desc {
                dst_image->resource.Get(),
                D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                D3D12CalcSubresource(
                    level,
                    layer,
                    0,
                    dst_image->resource_desc.MipLevels,
                    dst_image->resource_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ?
                        1 :
                        dst_image->resource_desc.DepthOrArraySize
                )
            };

            for (auto&& copy_region : new_regions) {
                const D3D12_TEXTURE_COPY_LOCATION src_desc {
                    src_buffer->resource.Get(),
                    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    copy_region.src_desc,
                };

                (*command_buffer)->CopyTextureRegion(
                    &dst_desc,
                    copy_region.dst_x,
                    copy_region.dst_y,
                    copy_region.dst_z,
                    &src_desc,
                    &copy_region.box
                );
            }
        }
    }

    auto const num_cs_regions { cs_regions.size() };
    if (!num_cs_regions) {
        return;
    }

    command_buffer->active_slot = std::nullopt;

    // Slow path starts here ->
    ComPtr<ID3D12DescriptorHeap> temp_heap { nullptr };
    const UINT handle_size {
        command_buffer->device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    };
    {
        const D3D12_DESCRIPTOR_HEAP_DESC desc {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            static_cast<UINT>(num_cs_regions),
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            0u
        };
        auto const hr { command_buffer->device()->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&temp_heap)) };
    }

    command_buffer->temp_heaps.push_back(temp_heap);

    const auto start_cpu = temp_heap->GetCPUDescriptorHandleForHeapStart();
    const auto start_gpu = temp_heap->GetGPUDescriptorHandleForHeapStart();

    (*command_buffer)->SetPipelineState(command_buffer->_device->pso_buffer_to_image.Get());
    (*command_buffer)->SetComputeRootSignature(command_buffer->_device->signature_buffer_to_image.Get());
    std::array<ID3D12DescriptorHeap *const, 1> heaps { temp_heap.Get() };
    (*command_buffer)->SetDescriptorHeaps(1, &heaps[0]);

    for (auto i : range(cs_regions.size())) {
        auto const& region { cs_regions[i] };

        // TODO: formats which don't support typed UAV ..
        // TODO: formats with different texel size..
        // TODO: 3D textures..
        // TODO: depth stencil format requires graphics pipeline .........

        std::array<uint32_t, 3> constant_data {
            region.offset,
            region.row_pitch,
            region.slice_pitch
        };
        (*command_buffer)->SetComputeRoot32BitConstants(
            2,
            static_cast<UINT>(constant_data.size()),
            constant_data.data(),
            0
        );

        D3D12_CPU_DESCRIPTOR_HANDLE cur_uav { start_cpu.ptr + i * handle_size };

        if (region.bits_format == 32) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc {
                DXGI_FORMAT_R32_UINT,
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY
            };
            uav_desc.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { region.mip_level, region.array_layer, 1, 0 };
            command_buffer->device()->CreateUnorderedAccessView(
                dst_image->resource.Get(),
                nullptr,
                &uav_desc,
                cur_uav
            );
        } else {
            WARN("vkCmdCopyBufferToImage: unhandled CS slow path copy with bit size {}", region.bits_format);
        }

        (*command_buffer)->SetComputeRootShaderResourceView(0, src_buffer->resource->GetGPUVirtualAddress());
        (*command_buffer)->SetComputeRootDescriptorTable(
            1,
            D3D12_GPU_DESCRIPTOR_HANDLE { start_gpu.ptr + i * handle_size }
        );

        (*command_buffer)->Dispatch(
            region.extent.width,
            region.extent.height,
            region.extent.depth
        );
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(
    VkCommandBuffer                             _commandBuffer,
    VkImage                                     _srcImage,
    VkImageLayout                               srcImageLayout,
    VkBuffer                                    _dstBuffer,
    uint32_t                                    regionCount,
    const VkBufferImageCopy*                    pRegions
) {
    TRACE("vkCmdCopyImageToBuffer");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto src_image { reinterpret_cast<image_t *>(_srcImage) };
    auto dst_buffer { reinterpret_cast<buffer_t *>(_dstBuffer) };
    auto regions { span<const VkBufferImageCopy>(pRegions, regionCount) };

    const auto img_width { src_image->resource_desc.Width };
    const auto img_height { src_image->resource_desc.Height };
    const auto img_depth {
        src_image->resource_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D ?
            1 :
            src_image->resource_desc.DepthOrArraySize
    };

    for (auto const& region : regions) {
        const auto level { region.imageSubresource.mipLevel };
        const auto width { std::max(static_cast<UINT>(img_width >> level), static_cast<UINT>(1u)) };
        const auto height { std::max(static_cast<UINT>(img_height >> level), static_cast<UINT>(1u)) };
        const auto depth { std::max(static_cast<UINT>(img_depth >> level), static_cast<UINT>(1u)) };

        const auto base_layer { region.imageSubresource.baseArrayLayer };
        const auto num_layers { region.imageSubresource.layerCount };

        for (auto layer : range(base_layer, base_layer+num_layers)) {
            const auto buffer_width {
                region.bufferRowLength ? region.bufferRowLength : region.imageExtent.width
            };
            const auto buffer_height {
                region.bufferImageHeight ? region.bufferImageHeight : region.imageExtent.height
            };

            const auto byte_per_texel { src_image->block_data.bits / 8 };
            const auto row_pitch {
                (up_align(buffer_width, src_image->block_data.width) / src_image->block_data.width) * byte_per_texel
            };

            // TODO: alignment?
            const D3D12_TEXTURE_COPY_LOCATION dst_desc {
                dst_buffer->resource.Get(),
                D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    region.bufferOffset,
                    D3D12_SUBRESOURCE_FOOTPRINT {
                        src_image->resource_desc.Format,
                        width,
                        height,
                        depth,
                        row_pitch,
                    },
                },
            };

            const D3D12_TEXTURE_COPY_LOCATION src_desc {
                src_image->resource.Get(),
                D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                D3D12CalcSubresource(
                    level,
                    layer,
                    0,
                    src_image->resource_desc.MipLevels,
                    src_image->resource_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ?
                        1 :
                        src_image->resource_desc.DepthOrArraySize
                ),
            };

            const D3D12_BOX box {
                0, 0, 0,
                up_align(region.imageExtent.width, src_image->block_data.width),
                up_align(region.imageExtent.height, src_image->block_data.height),
                region.imageExtent.depth,
            };

            (*command_buffer)->CopyTextureRegion(
                &dst_desc,
                region.imageOffset.x,
                region.imageOffset.y,
                region.imageOffset.z,
                &src_desc,
                &box
            );
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdUpdateBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                dataSize,
    const void*                                 pData
) {
    TRACE("vkCmdUpdateBuffer");
    WARN("vkCmdUpdateBuffer unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                size,
    uint32_t                                    data
) {
    TRACE("vkCmdFillBuffer");
    WARN("vkCmdFillBuffer unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     image,
    VkImageLayout                               imageLayout,
    const VkClearColorValue*                    pColor,
    uint32_t                                    rangeCount,
    const VkImageSubresourceRange*              pRanges
) {
    TRACE("vkCmdClearColorImage");
    WARN("vkCmdClearColorImage unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdClearDepthStencilImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     image,
    VkImageLayout                               imageLayout,
    const VkClearDepthStencilValue*             pDepthStencil,
    uint32_t                                    rangeCount,
    const VkImageSubresourceRange*              pRanges
) {
    TRACE("vkCmdClearDepthStencilImage");
    WARN("vkCmdClearDepthStencilImage unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdClearAttachments(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    attachmentCount,
    const VkClearAttachment*                    pAttachments,
    uint32_t                                    rectCount,
    const VkClearRect*                          pRects
) {
    TRACE("vkCmdClearAttachments");
    WARN("vkCmdClearAttachments unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdResolveImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     srcImage,
    VkImageLayout                               srcImageLayout,
    VkImage                                     dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkImageResolve*                       pRegions
) {
    TRACE("vkCmdResolveImage");
    WARN("vkCmdResolveImage unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     event,
    VkPipelineStageFlags                        stageMask
) {
    TRACE("");
    WARN("unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     event,
    VkPipelineStageFlags                        stageMask
) {
    TRACE("");
    WARN("unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    eventCount,
    const VkEvent*                              pEvents,
    VkPipelineStageFlags                        srcStageMask,
    VkPipelineStageFlags                        dstStageMask,
    uint32_t                                    memoryBarrierCount,
    const VkMemoryBarrier*                      pMemoryBarriers,
    uint32_t                                    bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier*                pBufferMemoryBarriers,
    uint32_t                                    imageMemoryBarrierCount,
    const VkImageMemoryBarrier*                 pImageMemoryBarriers
) {
    TRACE("");
    WARN("unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlags                        srcStageMask,
    VkPipelineStageFlags                        dstStageMask,
    VkDependencyFlags                           dependencyFlags,
    uint32_t                                    memoryBarrierCount,
    const VkMemoryBarrier*                      pMemoryBarriers,
    uint32_t                                    bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier*                pBufferMemoryBarriers,
    uint32_t                                    imageMemoryBarrierCount,
    const VkImageMemoryBarrier*                 pImageMemoryBarriers
) {
    TRACE("vkCmdPipelineBarrier");
    WARN("vkCmdPipelineBarrier unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdBeginQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    VkQueryControlFlags                         flags
) {
    TRACE("");
    WARN("unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdEndQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query
) {
    TRACE("");
    WARN("unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdResetQueryPool(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount
) {
    TRACE("vkCmdResetQueryPool");
    WARN("vkCmdResetQueryPool unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlagBits                     pipelineStage,
    VkQueryPool                                 queryPool,
    uint32_t                                    query
) {
    TRACE("vkCmdWriteTimestamp");
    WARN("vkCmdWriteTimestamp unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyQueryPoolResults(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                stride,
    VkQueryResultFlags                          flags
) {
    TRACE("vkCmdCopyQueryPoolResults");
    WARN("vkCmdCopyQueryPoolResults unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(
    VkCommandBuffer                             _commandBuffer,
    VkPipelineLayout                            _layout,
    VkShaderStageFlags                          stageFlags,
    uint32_t                                    offset,
    uint32_t                                    size,
    const void*                                 pValues
) {
    TRACE("vkCmdPushConstants");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto layout { reinterpret_cast<pipeline_layout_t *>(_layout) };
    auto const& root_constants { layout->root_constants };
    auto values { span<const uint32_t>(static_cast<const uint32_t *>(pValues), size) };

    auto start_slot { 0 };

    for (auto const& root_constant : root_constants) {
        if (root_constant.offset < offset + size &&
            offset < (root_constant.offset + root_constant.size))
        {
            const auto start { root_constant.offset - offset };
            const auto end { root_constant.offset + root_constant.size - offset };

            for (auto i : range(start / 4, end / 4)) {
                if (stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
                    command_buffer->compute_slot.root_data.set_constant(start_slot + i, values[i]);
                }
                if (stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
                    command_buffer->graphics_slot.root_data.set_constant(start_slot + i, values[i]);
                }
            }
        }

        start_slot += root_constant.size / 4;
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(
    VkCommandBuffer                             _commandBuffer,
    const VkRenderPassBeginInfo*                pRenderPassBegin,
    VkSubpassContents                           contents
) {
    TRACE("vkCmdBeginRenderPass");
    WARN("vkCmdBeginRenderPass unimplemented");

    auto const& info { *pRenderPassBegin };
    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto render_pass { reinterpret_cast<render_pass_t *>(info.renderPass) };
    auto framebuffer { reinterpret_cast<framebuffer_t *>(info.framebuffer) };
    std::vector<VkClearValue> clear_values(info.pClearValues, info.pClearValues + info.clearValueCount);

    command_buffer->pass_cache = command_buffer_t::pass_cache_t {
        0,
        render_pass,
        framebuffer,
        clear_values,
        D3D12_RECT {
            static_cast<LONG>(info.renderArea.offset.x),
            static_cast<LONG>(info.renderArea.offset.y),
            static_cast<LONG>(info.renderArea.offset.x + info.renderArea.extent.width),
            static_cast<LONG>(info.renderArea.offset.y + info.renderArea.extent.height)
        }
    };

    command_buffer->begin_subpass(contents);
}

VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass(
    VkCommandBuffer                             _commandBuffer,
    VkSubpassContents                           contents
) {
    TRACE("vkCmdNextSubpass");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };

    command_buffer->end_subpass();
    command_buffer->pass_cache->subpass += 1;
    command_buffer->begin_subpass(contents);
}

VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass(
    VkCommandBuffer                             _commandBuffer
) {
    TRACE("vkCmdEndRenderPass");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };

    command_buffer->end_subpass();
    command_buffer->pass_cache = std::nullopt;
}

VKAPI_ATTR void VKAPI_CALL vkCmdExecuteCommands(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    commandBufferCount,
    const VkCommandBuffer*                      pCommandBuffers
) {
    TRACE("vkCmdExecuteCommands");
    WARN("vkCmdExecuteCommands unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(
    VkDevice                                    _device,
    const VkSwapchainCreateInfoKHR*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSwapchainKHR*                             pSwapchain
) {
    TRACE("vkCreateSwapchainKHR");

    auto const& info { *pCreateInfo };
    auto device { reinterpret_cast<device_t *>(_device) };
    auto surface { reinterpret_cast<surface_t *>(info.surface) };
    auto old_swapchain { reinterpret_cast<swapchain_t *>(info.oldSwapchain) };

    if (old_swapchain) {
        for (auto& image : old_swapchain->images) {
            image.resource.Reset();
        }
        old_swapchain->swapchain.Reset();
    }

    auto vk_format = info.imageFormat;
    // Flip model swapchain doesn't allow srgb image formats, requires workaround with UNORM.
    switch (vk_format) {
        case VK_FORMAT_B8G8R8A8_SRGB: vk_format = VK_FORMAT_B8G8R8A8_UNORM; break;
        case VK_FORMAT_R8G8B8A8_SRGB: vk_format = VK_FORMAT_R8G8B8A8_UNORM; break;
    }
    auto const format { formats[vk_format] };
    auto swapchain { new swapchain_t };

    ComPtr<IDXGISwapChain1> dxgi_swapchain { nullptr };
    const DXGI_SWAP_CHAIN_DESC1 desc {
        info.imageExtent.width,
        info.imageExtent.height,
        format,
        FALSE,
        { 1, 0 },
        0, // TODO: usage
        info.minImageCount,
        DXGI_SCALING_NONE,
        DXGI_SWAP_EFFECT_FLIP_DISCARD, // TODO
        DXGI_ALPHA_MODE_UNSPECIFIED, // TODO
        0, // TODO: flags
    };

    {
        const auto hr {
            surface->dxgi_factory->CreateSwapChainForHwnd(
                device->present_queue.queue.Get(),
                surface->hwnd,
                &desc,
                nullptr, // TODO: fullscreen
                nullptr, // TODO: restrict?
                &dxgi_swapchain
            )
        };

        if (FAILED(hr)) {
            assert(false);
        }
        // TODO: errror
    }

    if (FAILED(dxgi_swapchain.As<IDXGISwapChain3>(&(swapchain->swapchain)))) {
        ERR("Couldn't convert swapchain to `IDXGISwapChain3`!");
    }

    for (auto i : range(info.minImageCount)) {
        ComPtr<ID3D12Resource> resource { nullptr };
        swapchain->swapchain->GetBuffer(i, IID_PPV_ARGS(&resource));

        const D3D12_RESOURCE_DESC desc {
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            0,
            info.imageExtent.width,
            info.imageExtent.height,
            1,
            1,
            format,
            { 1, 0 },
            D3D12_TEXTURE_LAYOUT_UNKNOWN, // TODO
            D3D12_RESOURCE_FLAG_NONE, // TODO
        };

        swapchain->images.emplace_back(
            image_t {
                resource,
                { },
                desc,
                formats_block[vk_format],
                info.imageUsage,
            }
        );
    }

    *pSwapchain = reinterpret_cast<VkSwapchainKHR>(swapchain);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    const VkAllocationCallbacks*                pAllocator
) {
    WARN("vkDestroySwapchainKHR unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              _swapchain,
    uint32_t*                                   pSwapchainImageCount,
    VkImage*                                    pSwapchainImages
) {
    TRACE("vkGetSwapchainImagesKHR");

    auto swapchain { reinterpret_cast<swapchain_t *>(_swapchain) };
    auto num_images { static_cast<uint32_t>(swapchain->images.size()) };

    if (!pSwapchainImages) {
        *pSwapchainImageCount = num_images;
        return VK_SUCCESS;
    }

    auto result { VK_SUCCESS };
    if (*pSwapchainImageCount < num_images) {
        num_images = *pSwapchainImageCount;
        result = VK_INCOMPLETE;
    }

    for (auto i : range(num_images)) {
        pSwapchainImages[i] = reinterpret_cast<VkImage>(&swapchain->images[i]);
    }

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImageKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              _swapchain,
    uint64_t                                    timeout,
    VkSemaphore                                 semaphore,
    VkFence                                     fence,
    uint32_t*                                   pImageIndex
) {
    TRACE("vkAcquireNextImageKHR");

    // TODO: single in order images atm with additional limitations, looking into alternatives..
    // TODO: sync stuff

    auto swapchain { reinterpret_cast<swapchain_t *>(_swapchain) };
    *pImageIndex = swapchain->swapchain->GetCurrentBackBufferIndex();

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(
    VkQueue                                     queue,
    const VkPresentInfoKHR*                     pPresentInfo
) {
    TRACE("vkQueuePresentKHR");

    auto const& info { *pPresentInfo };
    auto wait_semaphores { span<const VkSemaphore>(info.pWaitSemaphores, info.waitSemaphoreCount) };
    auto swapchains { span<const VkSwapchainKHR>(info.pSwapchains, info.swapchainCount) };
    auto image_indices { span<const uint32_t>(info.pImageIndices, info.swapchainCount) };

    // TODO: image indices, results, semaphores

    for (auto i : range(info.swapchainCount)) {
        auto const& swapchain { reinterpret_cast<swapchain_t *>(swapchains[i]) };

        swapchain->swapchain->Present(0, 0); // TODO: vsync?
    }

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySurfaceKHR(
    VkInstance                                  instance,
    VkSurfaceKHR                                surface,
    const VkAllocationCallbacks*                pAllocator
) {
    TRACE("vkDestroySurfaceKHR unimplemented");
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    VkSurfaceKHR                                surface,
    VkBool32*                                   pSupported
) {
    TRACE("vkGetPhysicalDeviceSurfaceSupportKHR");

    *pSupported = queueFamilyIndex == QUEUE_FAMILY_GENERAL_PRESENT;

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                _surface,
    VkSurfaceCapabilitiesKHR*                   pSurfaceCapabilities
) {
    TRACE("vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    auto surface { reinterpret_cast<surface_t *>(_surface) };

    RECT rect;
    if (!::GetClientRect(surface->hwnd, &rect)) {
        // TODO
        ERR("Couldn't get size of window");
    }

    const auto width { static_cast<uint32_t>(rect.right - rect.left) };
    const auto height { static_cast<uint32_t>(rect.bottom - rect.top) };

    *pSurfaceCapabilities = {
        // Image count due to FLIP_DISCARD
        2, // minImageCount
        16, // maxImageCount
        { width, height }, // currentExtent
        { width, height }, //minImageExtent
        { width, height }, // maxImageExtent
        1, // maxImageArrayLayers // TODO
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, // supportedTransforms // TODO
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, // currentTransform // TODO
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, // supportedCompositeAlpha // TODO
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, // supportedUsageFlags // TODO
    };

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceFormatsKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pSurfaceFormatCount,
    VkSurfaceFormatKHR*                         pSurfaceFormats
) {
    TRACE("vkGetPhysicalDeviceSurfaceFormatsKHR");

    // TODO: more formats
    const std::array<VkSurfaceFormatKHR, 2> formats {{
        { VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }
    }};

    auto num_formats { static_cast<uint32_t>(formats.size()) };

    if (!pSurfaceFormats) {
        *pSurfaceFormatCount = num_formats;
        return VK_SUCCESS;
    }

    auto result { VK_SUCCESS };
    if (*pSurfaceFormatCount < num_formats) {
        num_formats = *pSurfaceFormatCount;
        result = VK_INCOMPLETE;
    }

    for (auto i : range(num_formats)) {
        pSurfaceFormats[i] = formats[i];
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfacePresentModesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pPresentModeCount,
    VkPresentModeKHR*                           pPresentModes
) {
    TRACE("vkGetPhysicalDeviceSurfacePresentModesKHR");

    // TODO
    const std::array<VkPresentModeKHR, 1> modes {
        VK_PRESENT_MODE_FIFO_KHR
    };

    auto num_modes { static_cast<uint32_t>(modes.size()) };

    if (!pPresentModes) {
        *pPresentModeCount = num_modes;
        return VK_SUCCESS;
    }

    auto result { VK_SUCCESS };
    if (*pPresentModeCount < num_modes) {
        num_modes = *pPresentModeCount;
        result = VK_INCOMPLETE;
    }

    for (auto i : range(num_modes)) {
        pPresentModes[i] = modes[i];
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateWin32SurfaceKHR(
    VkInstance                                  _instance,
    const VkWin32SurfaceCreateInfoKHR*          pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface
) {
    TRACE("vkCreateWin32SurfaceKHR unimplemented");

    // TODO: multiple surfaces?
    auto instance { reinterpret_cast<instance_t *>(_instance) };
    auto const& info { *pCreateInfo };

    *pSurface = reinterpret_cast<VkSurfaceKHR>(
        new surface_t { instance->dxgi_factory, info.hwnd}
    );

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2KHR(
    VkPhysicalDevice                            _physicalDevice,
    VkPhysicalDeviceFeatures2KHR*               pFeatures
) {
    TRACE("vkGetPhysicalDeviceFeatures2KHR");

    vkGetPhysicalDeviceFeatures(_physicalDevice, &pFeatures->features);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2KHR(
    VkPhysicalDevice                            _physicalDevice,
    VkPhysicalDeviceProperties2KHR*             pProperties
) {
    TRACE("vkGetPhysicalDeviceProperties2KHR");

    auto physical_device { reinterpret_cast<physical_device_t *>(_physicalDevice) };

    auto next { static_cast<VkStructureType *>(pProperties->pNext) };
    while (next) {
        switch (*next) {
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT: {
                if (physical_device->conservative_properties) {
                    const auto& conservative { *physical_device->conservative_properties };
                    auto& properties { *reinterpret_cast<VkPhysicalDeviceConservativeRasterizationPropertiesEXT *>(next) };

                    properties.primitiveOverestimationSize                  = conservative.primitiveOverestimationSize;
                    properties.maxExtraPrimitiveOverestimationSize          = conservative.maxExtraPrimitiveOverestimationSize;
                    properties.extraPrimitiveOverestimationSizeGranularity  = conservative.extraPrimitiveOverestimationSizeGranularity;
                    properties.primitiveUnderestimation                     = conservative.primitiveUnderestimation;
                    properties.conservativePointAndLineRasterization        = conservative.conservativePointAndLineRasterization;
                    properties.degenerateTrianglesRasterized                = conservative.degenerateTrianglesRasterized;
                    properties.degenerateLinesRasterized                    = conservative.degenerateLinesRasterized;
                    properties.fullyCoveredFragmentShaderInputVariable      = conservative.fullyCoveredFragmentShaderInputVariable;
                    properties.conservativeRasterizationPostDepthCoverage   = conservative.conservativeRasterizationPostDepthCoverage;
                }
            } break;
        }
        next = static_cast<VkStructureType *>(next++);
    }

    vkGetPhysicalDeviceProperties(_physicalDevice, &pProperties->properties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2KHR(
    VkPhysicalDevice                            _physicalDevice,
    VkFormat                                    format,
    VkFormatProperties2KHR*                     pFormatProperties
) {
    TRACE("vkGetPhysicalDeviceFormatProperties2KHR");

    vkGetPhysicalDeviceFormatProperties(_physicalDevice, format, &pFormatProperties->formatProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties2KHR(
    VkPhysicalDevice                            _physicalDevice,
    const VkPhysicalDeviceImageFormatInfo2KHR*  pImageFormatInfo,
    VkImageFormatProperties2KHR*                pImageFormatProperties
) {
    TRACE("vkGetPhysicalDeviceImageFormatProperties2KHR");

    auto const& info { *pImageFormatInfo };

    const auto result {
        vkGetPhysicalDeviceImageFormatProperties(
            _physicalDevice,
            info.format,
            info.type,
            info.tiling,
            info.usage,
            info.flags,
            &pImageFormatProperties->imageFormatProperties
        )
    };
    return result;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2KHR(
    VkPhysicalDevice                            _physicalDevice,
    uint32_t*                                   pQueueFamilyPropertyCount,
    VkQueueFamilyProperties2KHR*                pQueueFamilyProperties
) {
    TRACE("vkGetPhysicalDeviceQueueFamilyProperties2KHR");

    vkGetPhysicalDeviceQueueFamilyProperties(
        _physicalDevice,
        pQueueFamilyPropertyCount,
        pQueueFamilyProperties ?
            &pQueueFamilyProperties->queueFamilyProperties :
            nullptr
    );
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2KHR(
    VkPhysicalDevice                            _physicalDevice,
    VkPhysicalDeviceMemoryProperties2KHR*       pMemoryProperties
) {
    TRACE("vkGetPhysicalDeviceMemoryProperties2KHR");

    vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &pMemoryProperties->memoryProperties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties2KHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceSparseImageFormatInfo2KHR* pFormatInfo,
    uint32_t*                                   pPropertyCount,
    VkSparseImageFormatProperties2KHR*          pProperties
) {
    TRACE("vkGetPhysicalDeviceSparseImageFormatProperties2KHR");
    WARN("vkGetPhysicalDeviceSparseImageFormatProperties2KHR unimplemented");
}