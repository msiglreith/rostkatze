#pragma once

#include "impl.hpp"

#include <shared_mutex>

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
