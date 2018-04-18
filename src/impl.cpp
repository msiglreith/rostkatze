
#include "icd.hpp"
#include "impl.hpp"
#include "descriptors_cpu.hpp"
#include "descriptors_gpu.hpp"
#include "descriptors_virtual.hpp"
#include "command_list.hpp"

#include <vk_icd.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include <gsl/gsl>
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

enum root_signature_spaces {
    PUSH_CONSTANT_REGISTER_SPACE = 0,
    DYNAMIC_OFFSET_SPACE,
    DESCRIPTOR_TABLE_INITIAL_SPACE,
};

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

// Header part of a vulkan struct
struct vulkan_struct_t {
    VkStructureType type;
    vulkan_struct_t *next;
};

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

    // TODO: take other parts into account
    static constexpr VkDeviceSize max_resource_size =
        D3D12_REQ_RESOURCE_SIZE_IN_MEGABYTES_EXPRESSION_A_TERM << 20;

    // TODO: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    *pImageFormatProperties = VkImageFormatProperties {
        VkExtent3D {
            D3D12_REQ_TEXTURE1D_U_DIMENSION,
            D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION,
            D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
        },
        D3D12_REQ_MIP_LEVELS,
        1,
        VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT,
        max_resource_size,
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

    auto physical_device { reinterpret_cast<physical_device_t *>(_physicalDevice) };

    auto const& info { *pCreateInfo };
    span<const VkDeviceQueueCreateInfo> queue_infos {
        info.pQueueCreateInfos,
        static_cast<int32_t>(info.queueCreateInfoCount),
    };

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
            command_lists[i] = command_buffer->raw_command_list();
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
    if (info.usage & (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
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
    if (format == DXGI_FORMAT_UNKNOWN) {
        ERR("Unsupported image format {}", info.format);
    }

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

        auto conservative_rasterization { D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF };

        auto next { static_cast<const vulkan_struct_t *>(rasterization_state.pNext) };
        for(; next; next = next->next) {
            switch (next->type) {
                case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT: {
                    auto const& conservative {
                        *reinterpret_cast<const VkPipelineRasterizationConservativeStateCreateInfoEXT *>(next)
                    };
                    switch (conservative.conservativeRasterizationMode) {
                        case VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT:
                            conservative_rasterization = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
                            break;
                        case VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT:
                        case VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT:
                            conservative_rasterization = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
                            break;
                    }
                } break;
            }
        }

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
            conservative_rasterization
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

    // Collect number of specified descriptor tables and dynamic offset root constants
    // in the create root signature
    auto num_descriptor_ranges { 0u };
    auto num_dynamic_offsets { 0u };
    for (auto const& _set : set_layouts) {
        auto set { reinterpret_cast<descriptor_set_layout_t *>(_set) };
        for (auto const& layout : set->layouts) {
            num_descriptor_ranges += layout.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? 2 : 1;

            if (
                layout.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                layout.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
            ) {
                // Dynamic buffers have an additional root constant containing the dynamic offset
                D3D12_ROOT_PARAMETER parameter {
                    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    { },
                    D3D12_SHADER_VISIBILITY_ALL // TODO
                };
                parameter.Constants = D3D12_ROOT_CONSTANTS {
                    num_dynamic_offsets,
                    DYNAMIC_OFFSET_SPACE,
                    layout.descriptor_count
                };

                num_dynamic_offsets += layout.descriptor_count;
            }
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
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                    type = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                    break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                    type = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
                    break;
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
            num_dynamic_offsets,
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
        info.anisotropyEnable ?
            D3D12_ENCODE_ANISOTROPIC_FILTER(reduction) :
            D3D12_ENCODE_BASIC_FILTER(
                base_filter(info.minFilter),
                base_filter(info.magFilter),
                mip_filter(info.mipmapMode),
                reduction
            )
    };

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

    pool->slice_sampler.start = std::make_tuple(device->descriptors_gpu_sampler.alloc(num_sampler), D3D12_GPU_DESCRIPTOR_HANDLE { 0 });
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
    const auto handle_size_sampler_cpu { device->descriptors_gpu_sampler.handle_size() };

    for (auto i : range(info.descriptorSetCount)) {
        auto layout { reinterpret_cast<descriptor_set_layout_t *>(layouts[i]) };
        auto descriptor_set = new descriptor_set_t;

        // TODO: precompute once
        auto num_cbv_srv_uav { 0u };
        auto num_sampler { 0u };
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

        descriptor_set->set_cbv_srv_uav = num_cbv_srv_uav ?
            std::optional<descriptor_set_placed_t>({ descriptor_gpu_cbv_srv_uav }) :
            std::nullopt;
        descriptor_set->set_sampler = num_sampler ?
            std::optional<descriptor_set_virtual_t>({ descriptor_cpu_sampler, num_sampler }) : // Samplers live in cpu heaps
            std::nullopt;

        if (num_sampler) {
            device->descriptors_gpu_sampler.assign_set(descriptor_cpu_sampler, descriptor_set);
        }

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
                    descriptor_cpu_sampler.ptr += binding.descriptor_count * handle_size_sampler_cpu;
                } break;
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                    descriptor_cpu_sampler.ptr += binding.descriptor_count * handle_size_sampler_cpu;
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
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: {
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
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
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

        auto command_buffer {
            new command_buffer_t(
                pool.allocator.Get(),
                command_list,
                device,
                device->descriptors_gpu_cbv_srv_uav.heap(),
                device->descriptors_gpu_sampler.cpu_heap()
            )
        };
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

VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceWin32PresentationSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex
) {
    return queueFamilyIndex == QUEUE_FAMILY_GENERAL_PRESENT;
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

    auto next { static_cast<vulkan_struct_t *>(pProperties->pNext) };
    for(; next; next = next->next) {
        switch (next->type) {
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
