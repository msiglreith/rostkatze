#pragma once

#include "impl.hpp"
#include "device.hpp"

#include <stdx/hash.hpp>
#include <stdx/match.hpp>

enum class draw_type {
    DRAW,
    DRAW_INDEXED,
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

private:
    auto operator->() -> ID3D12GraphicsCommandList2* {
        return this->command_list.Get();
    }

// Native functionality
public:
    auto resolve_subresource(
        ID3D12Resource *pDstResource,
        UINT           DstSubresource,
        ID3D12Resource *pSrcResource,
        UINT           SrcSubresource,
        DXGI_FORMAT    Format
    ) {
        this->command_list->ResolveSubresource(
            pDstResource,
            DstSubresource,
            pSrcResource,
            SrcSubresource,
            Format
        );
    }

    auto cmd_set_descriptor_heaps(
        UINT                        NumDescriptorHeaps,
        ID3D12DescriptorHeap *const *ppDescriptorHeaps
    ) {
        this->command_list->SetDescriptorHeaps(
            NumDescriptorHeaps,
            ppDescriptorHeaps
        );
    }

    auto cmd_set_compute_root_signature(ID3D12RootSignature *pRootSignature) {
        this->command_list->SetComputeRootSignature(pRootSignature);
    }

    auto cmd_set_graphics_root_signature(ID3D12RootSignature *pRootSignature) {
        this->command_list->SetGraphicsRootSignature(pRootSignature);
    }

    auto cmd_set_compute_root_descriptor_table(
        UINT                        RootParameterIndex,
        D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor
    ) {
        this->command_list->SetComputeRootDescriptorTable(
            RootParameterIndex,
            BaseDescriptor
        );
    }

    auto cmd_set_compute_root_constants(
        UINT RootParameterIndex,
        UINT Num32BitValuesToSet,
        const void *pSrcData,
        UINT DestOffsetIn32BitValues
    ) {
        this->command_list->SetComputeRoot32BitConstants(
            RootParameterIndex,
            Num32BitValuesToSet,
            pSrcData,
            DestOffsetIn32BitValues
        );
    }

    auto cmd_set_compute_root_shader_resource_view(
        UINT                      RootParameterIndex,
        D3D12_GPU_VIRTUAL_ADDRESS BufferLocation
    ) {
       this->command_list->SetComputeRootShaderResourceView(
            RootParameterIndex,
            BufferLocation
       );
    }

    auto cmd_set_primitive_topolgy(D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopology) {
        this->command_list->IASetPrimitiveTopology(PrimitiveTopology);
    }

    auto cmd_set_scissors(UINT NumRects, const D3D12_RECT *pRects) {
        this->command_list->RSSetScissorRects(NumRects, pRects);
    }

    auto cmd_set_viewports(UINT NumViewports, const D3D12_VIEWPORT *pViewports) {
        this->command_list->RSSetViewports(NumViewports, pViewports);
    }

    auto cmd_set_blend_factor(const FLOAT BlendFactor[4]) {
        this->command_list->OMSetBlendFactor(BlendFactor);
    }

    auto cmd_set_stencil_ref(UINT StencilRef) {
        this->command_list->OMSetStencilRef(StencilRef);
    }

    auto cmd_set_depth_bounds(FLOAT Min, FLOAT Max) {
        this->command_list->OMSetDepthBounds(Min, Max);
    }

    auto cmd_set_index_buffer(const D3D12_INDEX_BUFFER_VIEW *pView) {
        this->command_list->IASetIndexBuffer(pView);
    }

    auto cmd_copy_buffer_region(
        ID3D12Resource *pDstBuffer,
        UINT64         DstOffset,
        ID3D12Resource *pSrcBuffer,
        UINT64         SrcOffset,
        UINT64         NumBytes
    ) {
        this->command_list->CopyBufferRegion(
            pDstBuffer,
            DstOffset,
            pSrcBuffer,
            SrcOffset,
            NumBytes
        );
    }

    auto cmd_copy_texture_region(
        const D3D12_TEXTURE_COPY_LOCATION *pDst,
        UINT                        DstX,
        UINT                        DstY,
        UINT                        DstZ,
        const D3D12_TEXTURE_COPY_LOCATION *pSrc,
        const D3D12_BOX                   *pSrcBox
    ) {
        this->command_list->CopyTextureRegion(
            pDst,
            DstX,
            DstY,
            DstZ,
            pSrc,
            pSrcBox
        );
    }

    auto cmd_set_pipeline_state(ID3D12PipelineState *pPipelineState) {
        this->command_list->SetPipelineState(pPipelineState);
    }

    auto cmd_dispatch(
        UINT ThreadGroupCountX,
        UINT ThreadGroupCountY,
        UINT ThreadGroupCountZ
    ) {
        this->command_list->Dispatch(
            ThreadGroupCountX,
            ThreadGroupCountY,
            ThreadGroupCountZ
        );
    }

    auto cmd_draw_instanced(
        UINT VertexCountPerInstance,
        UINT InstanceCount,
        UINT StartVertexLocation,
        UINT StartInstanceLocation
    ) {
        this->command_list->DrawInstanced(
            VertexCountPerInstance,
            InstanceCount,
            StartVertexLocation,
            StartInstanceLocation
        );
    }

    auto cmd_draw_indexed_instanced(
        UINT IndexCountPerInstance,
        UINT InstanceCount,
        UINT StartIndexLocation,
        INT  BaseVertexLocation,
        UINT StartInstanceLocation
    ) {
        this->command_list->DrawIndexedInstanced(
            IndexCountPerInstance,
            InstanceCount,
            StartIndexLocation,
            BaseVertexLocation,
            StartInstanceLocation
        );
    }

    auto cmd_execute_indirect(
        ID3D12CommandSignature *pCommandSignature,
        UINT                   MaxCommandCount,
        ID3D12Resource         *pArgumentBuffer,
        UINT64                 ArgumentBufferOffset,
        ID3D12Resource         *pCountBuffer,
        UINT64                 CountBufferOffset
    ) {
        this->command_list->ExecuteIndirect(
            pCommandSignature,
            MaxCommandCount,
            pArgumentBuffer,
            ArgumentBufferOffset,
            pCountBuffer,
            CountBufferOffset
        );
    }

    auto cmd_resource_barrier(
        UINT                   NumBarriers,
        const D3D12_RESOURCE_BARRIER *pBarriers
    ) {
        this->command_list->ResourceBarrier(NumBarriers, pBarriers);
    }

public:
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

            resolve_subresource(
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
