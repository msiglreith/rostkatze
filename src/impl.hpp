#pragma once

#include "icd.hpp"
#include "descriptors_cpu.hpp"
#include "descriptors_gpu.hpp"

#include <array>
#include <map>
#include <optional>
#include <shared_mutex>
#include <variant>
#include <vector>

#include <gsl/gsl>

#include <stdx/range.hpp>
#include <stdx/hash.hpp>

#include <wrl.h>
#include <d3d12.h>
#include <d3dx12.h>
#include <dxgi1_6.h>

#include <vk_icd.h>

using namespace Microsoft::WRL;
using namespace gsl;
using namespace stdx;

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

static const size_t MAX_VERTEX_BUFFER_SLOTS = 16;

static auto init_debug_interface() -> void {
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

struct physical_device_properties_t {
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t vendor_id;
    uint32_t device_id;
    VkPhysicalDeviceType device_type;
    std::array<char, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE> device_name;
};

struct physical_device_t {
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

struct semaphore_t {
    ComPtr<ID3D12Fence> fence;
};

struct surface_t {
    ComPtr<IDXGIFactory5> dxgi_factory;
    std::variant<HWND, IUnknown*> handle;
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

static auto up_align(UINT v, UINT alignment) -> UINT {
    return (v + alignment - 1) & ~(alignment - 1);
}

auto create_command_signature(
    ID3D12Device* device,
    D3D12_INDIRECT_ARGUMENT_TYPE type,
    UINT stride
) -> ComPtr<ID3D12CommandSignature>;
