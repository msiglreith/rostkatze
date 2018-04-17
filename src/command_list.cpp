#include <stdx/match.hpp>

#include "command_list.hpp"


using namespace stdx;

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
                    command_buffer->heap_gpu_cbv_srv_uav,
                    command_buffer->heap_gpu_sampler
                };
                command_buffer->command_recorder.cmd_set_descriptor_heaps(
                    command_buffer->heap_gpu_sampler ? 2 : 1,
                    &heaps[0]
                );
                command_buffer->active_slot = command_buffer_t::SLOT_GRAPHICS;
            }

            if (command_buffer->graphics_slot.signature != signature) {
                command_buffer->command_recorder.cmd_set_graphics_root_signature(signature);
                command_buffer->graphics_slot.signature = signature;
                // TODO: descriptor sets
            }
            command_buffer->graphics_slot.pipeline = pipeline;
            command_buffer->command_recorder.cmd_set_primitive_topolgy(pipeline->topology); // no need to cache this

            for (auto i : range(MAX_VERTEX_BUFFER_SLOTS)) {
                command_buffer->vertex_buffer_views[i].StrideInBytes = pipeline->vertex_strides[i];
            }

            // Apply static states
            if (pipeline->static_viewports) {
                command_buffer->command_recorder.cmd_set_viewports(
                    static_cast<UINT>(pipeline->static_viewports->size()),
                    pipeline->static_viewports->data()
                );
                command_buffer->viewports_dirty = false;
            }
            if (pipeline->static_scissors) {
                command_buffer->command_recorder.cmd_set_scissors(
                    static_cast<UINT>(pipeline->static_scissors->size()),
                    pipeline->static_scissors->data()
                );
                command_buffer->scissors_dirty = false;
            }
            if (pipeline->static_blend_factors) {
                command_buffer->command_recorder.cmd_set_blend_factor(
                    pipeline->static_blend_factors->factors
                );
            }
            if (pipeline->static_depth_bounds) {
                const auto [min, max] = *pipeline->static_depth_bounds;
                command_buffer->command_recorder.cmd_set_depth_bounds(min, max);
            }
            if (pipeline->static_stencil_reference) {
                command_buffer->command_recorder.cmd_set_stencil_ref(
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
                    command_buffer->heap_gpu_cbv_srv_uav,
                    command_buffer->heap_gpu_sampler
                };
                command_buffer->command_recorder.cmd_set_descriptor_heaps(
                    command_buffer->heap_gpu_sampler ? 2 : 1,
                    &heaps[0]
                );
                command_buffer->active_slot = command_buffer_t::SLOT_COMPUTE;
            }

            if (command_buffer->compute_slot.signature != signature) {
                command_buffer->command_recorder.cmd_set_compute_root_signature(signature);
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
    command_buffer->command_recorder.cmd_set_blend_factor(blendConstants);
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBounds(
    VkCommandBuffer                             _commandBuffer,
    float                                       minDepthBounds,
    float                                       maxDepthBounds
) {
    TRACE("vkCmdSetDepthBounds");

    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    command_buffer->command_recorder.cmd_set_depth_bounds(minDepthBounds, maxDepthBounds);
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
    command_buffer->command_recorder.cmd_set_stencil_ref(reference);
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

    // TODO: dynamic descriptors

    auto descriptor_sets { span<const VkDescriptorSet>(pDescriptorSets, descriptorSetCount) };
    auto command_buffer { reinterpret_cast<command_buffer_t *>(_commandBuffer) };
    auto layout { reinterpret_cast<pipeline_layout_t *>(_layout) };

    // CbvSrvUav descriptors have a fixed heap, we want to store only the offset to this heap
    const auto start_cbv_srv_uav {
        command_buffer->heap_gpu_cbv_srv_uav->GetGPUDescriptorHandleForHeapStart()
    };
    // Samplers live in a non-shader visible heap due to the size limitations.
    // We store the offset to the fixed size cpu heap and replace them on draw/dispatch.
    const auto start_sampler {
        command_buffer->heap_cpu_sampler->GetCPUDescriptorHandleForHeapStart()
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

            if (set->set_cbv_srv_uav) {
                // Only storing relative address, less storage needed
                const auto set_offset { set->set_cbv_srv_uav->gpu_start.ptr - start_cbv_srv_uav.ptr };
                pipeline.root_data.set_cbv_srv_uav(entry, static_cast<uint32_t>(set_offset));
                entry += 1;
            }
            if (set->set_sampler) {
                // Only storing relative address to the CPU heap, less storage needed
                const auto set_offset { set->set_sampler->cpu_handle.ptr - start_sampler.ptr };
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

    command_buffer->command_recorder.cmd_set_index_buffer(
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
        command_buffer->vertex_buffer_views[i].SizeInBytes = static_cast<UINT>(buffer->memory_requirements.size - offset);
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
    command_buffer->command_recorder.cmd_draw_instanced(
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
    command_buffer->command_recorder.cmd_draw_indexed_instanced(
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
    command_buffer->command_recorder.cmd_execute_indirect(
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
    command_buffer->command_recorder.cmd_execute_indirect(
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
    command_buffer->command_recorder.cmd_dispatch(
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
        command_buffer->command_recorder.cmd_copy_buffer_region(
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

            command_buffer->command_recorder.cmd_copy_texture_region(
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

    command_buffer->command_recorder.cmd_set_pipeline_state(command_buffer->_device->pso_blit_2d.Get());
    command_buffer->command_recorder.cmd_set_compute_root_signature(command_buffer->_device->signature_blit_2d.Get());
    std::array<ID3D12DescriptorHeap *const, 1> heaps { temp_heap.Get() };
    command_buffer->command_recorder.cmd_set_descriptor_heaps(1, &heaps[0]);

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


        command_buffer->command_recorder.cmd_set_compute_root_descriptor_table(
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
        command_buffer->command_recorder.cmd_set_compute_root_constants(
            1,
            static_cast<UINT>(constant_data.size()),
            constant_data.data(),
            0
        );

        command_buffer->command_recorder.cmd_dispatch(
            region.dstOffsets[1].x - region.dstOffsets[0].x,
            region.dstOffsets[1].y - region.dstOffsets[0].y,
            region.dstOffsets[1].z - region.dstOffsets[0].z
        );

        std::array<D3D12_RESOURCE_BARRIER, 2> uav_barriers {
            CD3DX12_RESOURCE_BARRIER::UAV(src_image->resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(dst_image->resource.Get())
        };
        command_buffer->command_recorder.cmd_resource_barrier(
            static_cast<UINT>(uav_barriers.size()),
            uav_barriers.data()
        );
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

    struct copy_region_t {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT src_desc;
        UINT dst_x;
        UINT dst_y;
        UINT dst_z;
        D3D12_BOX box;
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

            const auto image_offset_x { static_cast<UINT>(region.imageOffset.x) };
            const auto image_offset_y { static_cast<UINT>(region.imageOffset.y) };
            const auto image_offset_z { static_cast<UINT>(region.imageOffset.z) };

            // Aligning, in particular required for the case of block compressed formats and non-multiple width/height fields.
            const auto byte_per_texel { dst_image->block_data.bits / 8 };
            const auto num_rows { up_align(buffer_height, dst_image->block_data.height) / dst_image->block_data.height };
            const auto raw_row_pitch {
                (up_align(buffer_width, dst_image->block_data.width) / dst_image->block_data.width) * byte_per_texel
            };
            const auto raw_slice_pitch { raw_row_pitch * num_rows };
            const auto offset { region.bufferOffset + (layer - base_layer) * raw_slice_pitch * depth };
            const auto pitch_aligned { (raw_row_pitch & (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) == 0 };
            const auto offset_aligned { (offset & (D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1)) == 0 };

            // Store splitted copy segments
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
                            image_offset_x,
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
                        uint32_t offset,
                        uint32_t row_pitch,
                        uint32_t slice_pitch,
                        uint32_t texel_size,
                        uint32_t& texel_offset_x,
                        uint32_t& texel_offset_y,
                        uint32_t& texel_offset_z,
                        uint32_t block_width, uint32_t block_height
                    ) {
                        const uint32_t byte_offset_x = offset % row_pitch;
                        offset -= byte_offset_x;
                        const uint32_t byte_offset_y = offset % slice_pitch;
                        offset -= byte_offset_y;
                        const uint32_t byte_offset_z = offset;

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
                                image_offset_x,
                                image_offset_y,
                                image_offset_z,
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
                                    image_offset_x,
                                    image_offset_y,
                                    image_offset_z,
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
                                    image_offset_x + row_pitch_texels - texel_offset_x,
                                    image_offset_y,
                                    image_offset_z,
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
                    // End texture splitting by NXT
                }
            } else if (false) {
                // TODO: check if the format supports UAV
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
                // TODO: 3d textures
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
                            image_offset_x,
                            image_offset_y + y * dst_image->block_data.height,
                            image_offset_z,
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
                                image_offset_x + buffer_width - buffer_texels,
                                image_offset_y + y * dst_image->block_data.height,
                                image_offset_z,
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

            // Submit copy commands
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

                command_buffer->command_recorder.cmd_copy_texture_region(
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

    command_buffer->command_recorder.cmd_set_pipeline_state(
        command_buffer->_device->pso_buffer_to_image.Get()
    );
    command_buffer->command_recorder.cmd_set_compute_root_signature(
        command_buffer->_device->signature_buffer_to_image.Get()
    );
    std::array<ID3D12DescriptorHeap *const, 1> heaps { temp_heap.Get() };
    command_buffer->command_recorder.cmd_set_descriptor_heaps(1, &heaps[0]);

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
        command_buffer->command_recorder.cmd_set_compute_root_constants(
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

        command_buffer->command_recorder.cmd_set_compute_root_shader_resource_view(
            0, src_buffer->resource->GetGPUVirtualAddress()
        );
        command_buffer->command_recorder.cmd_set_compute_root_descriptor_table(
            1,
            D3D12_GPU_DESCRIPTOR_HANDLE { start_gpu.ptr + i * handle_size }
        );

        command_buffer->command_recorder.cmd_dispatch(
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

            command_buffer->command_recorder.cmd_copy_texture_region(
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
    TRACE("vkCmdSetEvent");
    WARN("unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     event,
    VkPipelineStageFlags                        stageMask
) {
    TRACE("vkCmdResetEvent");
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
    TRACE("vkCmdWaitEvents");
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
    TRACE("vkCmdBeginQuery");
    WARN("unimplemented");
}

VKAPI_ATTR void VKAPI_CALL vkCmdEndQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query
) {
    TRACE("vkCmdEndQuery");
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