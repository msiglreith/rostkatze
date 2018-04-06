#pragma once

#include "impl.hpp"

class command_buffer_recorder_t {
public:
    virtual ~command_buffer_recorder_t() { }

    virtual auto resolve_subresource(
        ID3D12Resource *pDstResource,
        UINT           DstSubresource,
        ID3D12Resource *pSrcResource,
        UINT           SrcSubresource,
        DXGI_FORMAT    Format
    ) -> void = 0;

    virtual auto cmd_set_descriptor_heaps(
        UINT                        NumDescriptorHeaps,
        ID3D12DescriptorHeap *const *ppDescriptorHeaps
    ) -> void = 0;

    virtual auto cmd_set_compute_root_signature(ID3D12RootSignature *pRootSignature) -> void = 0;
    virtual auto cmd_set_graphics_root_signature(ID3D12RootSignature *pRootSignature) -> void = 0;

    virtual auto cmd_set_compute_root_descriptor_table(
        UINT                        RootParameterIndex,
        D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor
    ) -> void = 0;

    virtual auto cmd_set_compute_root_constant(
        UINT RootParameterIndex,
        UINT SrcData,
        UINT DestOffsetIn32BitValues
    ) -> void = 0;

    virtual auto cmd_set_compute_root_constants(
        UINT RootParameterIndex,
        UINT Num32BitValuesToSet,
        const void *pSrcData,
        UINT DestOffsetIn32BitValues
    ) -> void = 0;

    virtual auto cmd_set_compute_root_shader_resource_view(
        UINT                      RootParameterIndex,
        D3D12_GPU_VIRTUAL_ADDRESS BufferLocation
    ) -> void = 0;

    virtual auto cmd_set_graphics_root_constant(
        UINT RootParameterIndex,
        UINT SrcData,
        UINT DestOffsetIn32BitValues
    ) -> void = 0;

    virtual auto cmd_set_graphics_root_descriptor_table(
        UINT                        RootParameterIndex,
        D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor
    ) -> void = 0;
    virtual auto cmd_clear_render_target_view(
        D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetView,
        const FLOAT                 ColorRGBA[4],
        UINT                        NumRects,
        const D3D12_RECT            *pRects
    ) -> void = 0;

    virtual auto cmd_clear_depth_stencil_view(
        D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView,
        D3D12_CLEAR_FLAGS           ClearFlags,
        FLOAT                       Depth,
        UINT8                       Stencil,
        UINT                        NumRects,
        const D3D12_RECT            *pRects
    ) -> void = 0;

    virtual auto cmd_set_render_targets(
        UINT                                NumRenderTargetDescriptors,
        const D3D12_CPU_DESCRIPTOR_HANDLE   *pRenderTargetDescriptors,
        BOOL                                RTsSingleHandleToDescriptorRange,
        const D3D12_CPU_DESCRIPTOR_HANDLE   *pDepthStencilDescriptor
    ) -> void = 0;

    virtual auto cmd_set_primitive_topolgy(D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopology) -> void = 0;
    virtual auto cmd_set_scissors(UINT NumRects, const D3D12_RECT *pRects) -> void = 0;
    virtual auto cmd_set_viewports(UINT NumViewports, const D3D12_VIEWPORT *pViewports) -> void = 0;
    virtual auto cmd_set_blend_factor(const FLOAT BlendFactor[4]) -> void = 0;
    virtual auto cmd_set_stencil_ref(UINT StencilRef) -> void = 0;
    virtual auto cmd_set_depth_bounds(FLOAT Min, FLOAT Max) -> void = 0;
    virtual auto cmd_set_index_buffer(const D3D12_INDEX_BUFFER_VIEW *pView) -> void = 0;

    virtual auto cmd_copy_buffer_region(
        ID3D12Resource *pDstBuffer,
        UINT64         DstOffset,
        ID3D12Resource *pSrcBuffer,
        UINT64         SrcOffset,
        UINT64         NumBytes
    ) -> void = 0;

    virtual auto cmd_copy_texture_region(
        const D3D12_TEXTURE_COPY_LOCATION *pDst,
        UINT                        DstX,
        UINT                        DstY,
        UINT                        DstZ,
        const D3D12_TEXTURE_COPY_LOCATION *pSrc,
        const D3D12_BOX                   *pSrcBox
    ) -> void = 0;

    virtual auto cmd_set_pipeline_state(ID3D12PipelineState *pPipelineState) -> void = 0;

    virtual auto cmd_dispatch(
        UINT ThreadGroupCountX,
        UINT ThreadGroupCountY,
        UINT ThreadGroupCountZ
    ) -> void = 0;

    virtual auto cmd_draw_instanced(
        UINT VertexCountPerInstance,
        UINT InstanceCount,
        UINT StartVertexLocation,
        UINT StartInstanceLocation
    ) -> void = 0;

    virtual auto cmd_draw_indexed_instanced(
        UINT IndexCountPerInstance,
        UINT InstanceCount,
        UINT StartIndexLocation,
        INT  BaseVertexLocation,
        UINT StartInstanceLocation
    ) -> void = 0;

    virtual auto cmd_execute_indirect(
        ID3D12CommandSignature *pCommandSignature,
        UINT                   MaxCommandCount,
        ID3D12Resource         *pArgumentBuffer,
        UINT64                 ArgumentBufferOffset,
        ID3D12Resource         *pCountBuffer,
        UINT64                 CountBufferOffset
    ) -> void = 0;

    virtual auto cmd_resource_barrier(
        UINT                   NumBarriers,
        const D3D12_RESOURCE_BARRIER *pBarriers
    ) -> void = 0;

    virtual auto cmd_set_vertex_buffers(
        UINT                     StartSlot,
        UINT                     NumViews,
        const D3D12_VERTEX_BUFFER_VIEW *pViews
    ) -> void = 0;
};

class command_buffer_recorder_native_t : public command_buffer_recorder_t {
public:
    explicit command_buffer_recorder_native_t(ID3D12GraphicsCommandList2* cmd_list) :
        command_list { cmd_list }
    {}

public:
    virtual auto resolve_subresource(
        ID3D12Resource *pDstResource,
        UINT           DstSubresource,
        ID3D12Resource *pSrcResource,
        UINT           SrcSubresource,
        DXGI_FORMAT    Format
    ) -> void override {
        this->command_list->ResolveSubresource(
            pDstResource,
            DstSubresource,
            pSrcResource,
            SrcSubresource,
            Format
        );
    }

    virtual auto cmd_set_descriptor_heaps(
        UINT                        NumDescriptorHeaps,
        ID3D12DescriptorHeap *const *ppDescriptorHeaps
    ) -> void override {
        this->command_list->SetDescriptorHeaps(
            NumDescriptorHeaps,
            ppDescriptorHeaps
        );
    }

    virtual auto cmd_set_compute_root_signature(ID3D12RootSignature *pRootSignature) -> void override {
        this->command_list->SetComputeRootSignature(pRootSignature);
    }

    virtual auto cmd_set_graphics_root_signature(ID3D12RootSignature *pRootSignature) -> void override {
        this->command_list->SetGraphicsRootSignature(pRootSignature);
    }

    virtual auto cmd_set_compute_root_descriptor_table(
        UINT                        RootParameterIndex,
        D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor
    ) -> void override {
        this->command_list->SetComputeRootDescriptorTable(
            RootParameterIndex,
            BaseDescriptor
        );
    }

    virtual auto cmd_set_compute_root_constant(
        UINT RootParameterIndex,
        UINT SrcData,
        UINT DestOffsetIn32BitValues
    ) -> void override {
        this->command_list->SetComputeRoot32BitConstant(
            RootParameterIndex,
            SrcData,
            DestOffsetIn32BitValues
        );
    }

    virtual auto cmd_set_compute_root_constants(
        UINT RootParameterIndex,
        UINT Num32BitValuesToSet,
        const void *pSrcData,
        UINT DestOffsetIn32BitValues
    ) -> void override {
        this->command_list->SetComputeRoot32BitConstants(
            RootParameterIndex,
            Num32BitValuesToSet,
            pSrcData,
            DestOffsetIn32BitValues
        );
    }

    virtual auto cmd_set_compute_root_shader_resource_view(
        UINT                      RootParameterIndex,
        D3D12_GPU_VIRTUAL_ADDRESS BufferLocation
    ) -> void override {
       this->command_list->SetComputeRootShaderResourceView(
            RootParameterIndex,
            BufferLocation
       );
    }

    virtual auto cmd_set_graphics_root_constant(
        UINT RootParameterIndex,
        UINT SrcData,
        UINT DestOffsetIn32BitValues
    ) -> void override {
        this->command_list->SetGraphicsRoot32BitConstant(
            RootParameterIndex,
            SrcData,
            DestOffsetIn32BitValues
        );
    }

    virtual auto cmd_set_graphics_root_descriptor_table(
        UINT                        RootParameterIndex,
        D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor
    ) -> void override {
        this->command_list->SetGraphicsRootDescriptorTable(
            RootParameterIndex,
            BaseDescriptor
        );
    }

    virtual auto cmd_clear_render_target_view(
        D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetView,
        const FLOAT                 ColorRGBA[4],
        UINT                        NumRects,
        const D3D12_RECT            *pRects
    ) -> void override {
        this->command_list->ClearRenderTargetView(
            RenderTargetView,
            ColorRGBA,
            NumRects,
            pRects
        );
    }

    virtual auto cmd_clear_depth_stencil_view(
        D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView,
        D3D12_CLEAR_FLAGS           ClearFlags,
        FLOAT                       Depth,
        UINT8                       Stencil,
        UINT                        NumRects,
        const D3D12_RECT            *pRects
    ) -> void override {
        this->command_list->ClearDepthStencilView(
            DepthStencilView,
            ClearFlags,
            Depth,
            Stencil,
            NumRects,
            pRects
        );
    }

    virtual auto cmd_set_render_targets(
        UINT                                NumRenderTargetDescriptors,
        const D3D12_CPU_DESCRIPTOR_HANDLE   *pRenderTargetDescriptors,
        BOOL                                RTsSingleHandleToDescriptorRange,
        const D3D12_CPU_DESCRIPTOR_HANDLE   *pDepthStencilDescriptor
    ) -> void override {
        this->command_list->OMSetRenderTargets(
            NumRenderTargetDescriptors,
            pRenderTargetDescriptors,
            RTsSingleHandleToDescriptorRange,
            pDepthStencilDescriptor
        );
    }

    virtual auto cmd_set_primitive_topolgy(D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopology) -> void override {
        this->command_list->IASetPrimitiveTopology(PrimitiveTopology);
    }

    virtual auto cmd_set_scissors(UINT NumRects, const D3D12_RECT *pRects) -> void override {
        this->command_list->RSSetScissorRects(NumRects, pRects);
    }

    virtual auto cmd_set_viewports(UINT NumViewports, const D3D12_VIEWPORT *pViewports) -> void override {
        this->command_list->RSSetViewports(NumViewports, pViewports);
    }

    virtual auto cmd_set_blend_factor(const FLOAT BlendFactor[4]) -> void override {
        this->command_list->OMSetBlendFactor(BlendFactor);
    }

    virtual auto cmd_set_stencil_ref(UINT StencilRef) -> void override {
        this->command_list->OMSetStencilRef(StencilRef);
    }

    virtual auto cmd_set_depth_bounds(FLOAT Min, FLOAT Max) -> void override {
        this->command_list->OMSetDepthBounds(Min, Max);
    }

    virtual auto cmd_set_index_buffer(const D3D12_INDEX_BUFFER_VIEW *pView) -> void override {
        this->command_list->IASetIndexBuffer(pView);
    }

    virtual auto cmd_copy_buffer_region(
        ID3D12Resource *pDstBuffer,
        UINT64         DstOffset,
        ID3D12Resource *pSrcBuffer,
        UINT64         SrcOffset,
        UINT64         NumBytes
    ) -> void override {
        this->command_list->CopyBufferRegion(
            pDstBuffer,
            DstOffset,
            pSrcBuffer,
            SrcOffset,
            NumBytes
        );
    }

    virtual auto cmd_copy_texture_region(
        const D3D12_TEXTURE_COPY_LOCATION *pDst,
        UINT                        DstX,
        UINT                        DstY,
        UINT                        DstZ,
        const D3D12_TEXTURE_COPY_LOCATION *pSrc,
        const D3D12_BOX                   *pSrcBox
    ) -> void override {
        this->command_list->CopyTextureRegion(
            pDst,
            DstX,
            DstY,
            DstZ,
            pSrc,
            pSrcBox
        );
    }

    virtual auto cmd_set_pipeline_state(ID3D12PipelineState *pPipelineState) -> void override {
        this->command_list->SetPipelineState(pPipelineState);
    }

    virtual auto cmd_dispatch(
        UINT ThreadGroupCountX,
        UINT ThreadGroupCountY,
        UINT ThreadGroupCountZ
    ) -> void override {
        this->command_list->Dispatch(
            ThreadGroupCountX,
            ThreadGroupCountY,
            ThreadGroupCountZ
        );
    }

    virtual auto cmd_draw_instanced(
        UINT VertexCountPerInstance,
        UINT InstanceCount,
        UINT StartVertexLocation,
        UINT StartInstanceLocation
    ) -> void override {
        this->command_list->DrawInstanced(
            VertexCountPerInstance,
            InstanceCount,
            StartVertexLocation,
            StartInstanceLocation
        );
    }

    virtual auto cmd_draw_indexed_instanced(
        UINT IndexCountPerInstance,
        UINT InstanceCount,
        UINT StartIndexLocation,
        INT  BaseVertexLocation,
        UINT StartInstanceLocation
    ) -> void override {
        this->command_list->DrawIndexedInstanced(
            IndexCountPerInstance,
            InstanceCount,
            StartIndexLocation,
            BaseVertexLocation,
            StartInstanceLocation
        );
    }

    virtual auto cmd_execute_indirect(
        ID3D12CommandSignature *pCommandSignature,
        UINT                   MaxCommandCount,
        ID3D12Resource         *pArgumentBuffer,
        UINT64                 ArgumentBufferOffset,
        ID3D12Resource         *pCountBuffer,
        UINT64                 CountBufferOffset
    ) -> void override {
        this->command_list->ExecuteIndirect(
            pCommandSignature,
            MaxCommandCount,
            pArgumentBuffer,
            ArgumentBufferOffset,
            pCountBuffer,
            CountBufferOffset
        );
    }

    virtual auto cmd_resource_barrier(
        UINT                   NumBarriers,
        const D3D12_RESOURCE_BARRIER *pBarriers
    ) -> void override {
        this->command_list->ResourceBarrier(NumBarriers, pBarriers);
    }

    virtual auto cmd_set_vertex_buffers(
        UINT                     StartSlot,
        UINT                     NumViews,
        const D3D12_VERTEX_BUFFER_VIEW *pViews
    ) -> void override {
        this->command_list->IASetVertexBuffers(
            StartSlot,
            NumViews,
            pViews
        );
    }

private:
    ID3D12GraphicsCommandList2* command_list;
};

class command_buffer_recorder_store_t : public command_buffer_recorder_t {
    enum class command_t {
        RESOLVE_SUBRESOURCE,
        SET_DESCRIPTOR_HEAPS,
        SET_COMPUTE_ROOT_SIGNATURE,
        SET_COMPUTE_ROOT_DESCRIPTOR_TABLE,
        SET_COMPUTE_ROOT_CONSTANT,
        SET_COMPUTE_ROOT_CONSTANTS,
        SET_COMPUTE_ROOT_SHADER_RESOURCE_VIEW,
        SET_GRAPHICS_ROOT_SIGNATURE,
        SET_GRAPHICS_ROOT_CONSTANT,
        SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE,
        CLEAR_RENDER_TARGET_VIEW,
        CLEAR_DEPTH_STENCIL_VIEW,
        SET_RENDER_TARGET,
        SET_PRIMITIVE_TOPOLOGY,
        SET_SCISSORS,
        SET_VIEWPORTS,
        SET_BLEND_FACTOR,
        SET_STENCIL_REF,
        SET_DEPTH_BOUNDS,
        SET_INDEX_BUFFER,
        COPY_BUFFER_REGION,
        COPY_TEXTURE_REGION,
        SET_PIPELINE_STATE,
        DISPATCH,
        DRAW_INSTANCED,
        DRAW_INDEXED_INSTANCED,
        EXECUTE_INDIRECT,
        RESOURCE_BARRIER,
        SET_VERTEX_BUFFERS,
    };

    struct resolve_subresource_t {
        ID3D12Resource *pDstResource;
        UINT           DstSubresource;
        ID3D12Resource *pSrcResource;
        UINT           SrcSubresource;
        DXGI_FORMAT    Format;
    };

    struct set_descriptor_heaps_t {
        UINT NumDescriptorHeaps;
        ID3D12DescriptorHeap const* DescriptorHeaps[2]; // Max 2 Heaps atm
    };

    struct set_compute_root_signature_t {
        ID3D12RootSignature *pRootSignature;
    };

    struct set_graphics_root_signature_t {
        ID3D12RootSignature *pRootSignature;
    };

    struct set_compute_root_descriptor_table_t {
        UINT                        RootParameterIndex;
        D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor;
    };

    struct set_graphics_root_descriptor_table_t {
        UINT                        RootParameterIndex;
        D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor;
    };

    struct set_compute_root_constant_t {
        UINT RootParameterIndex;
        UINT SrcData;
        UINT DestOffsetIn32BitValues;
    };

    struct set_graphics_root_constant_t {
        UINT RootParameterIndex;
        UINT SrcData;
        UINT DestOffsetIn32BitValues;
    };

    struct set_compute_root_shader_resource_view_t {
        UINT                      RootParameterIndex;
        D3D12_GPU_VIRTUAL_ADDRESS BufferLocation;
    };

    struct set_pipeline_state_t {
        ID3D12PipelineState *state;
    };

    struct set_primitive_topology_t {
        D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopology;
    };

    struct set_stencil_ref_t {
        UINT StencilRef;
    };

    struct set_depth_bounds_t {
        FLOAT Min;
        FLOAT Max;
    };

    struct dispatch_t {
        UINT ThreadGroupCountX;
        UINT ThreadGroupCountY;
        UINT ThreadGroupCountZ;
    };

    struct draw_instanced_t {
        UINT VertexCountPerInstance;
        UINT InstanceCount;
        UINT StartVertexLocation;
        UINT StartInstanceLocation;
    };

    struct draw_indexed_instanced_t {
        UINT IndexCountPerInstance;
        UINT InstanceCount;
        UINT StartIndexLocation;
        INT  BaseVertexLocation;
        UINT StartInstanceLocation;
    };

    struct execute_indirect_t {
        ID3D12CommandSignature *pCommandSignature;
        UINT                   MaxCommandCount;
        ID3D12Resource         *pArgumentBuffer;
        UINT64                 ArgumentBufferOffset;
        ID3D12Resource         *pCountBuffer;
        UINT64                 CountBufferOffset;
    };

public:
    virtual auto resolve_subresource(
        ID3D12Resource *pDstResource,
        UINT           DstSubresource,
        ID3D12Resource *pSrcResource,
        UINT           SrcSubresource,
        DXGI_FORMAT    Format
    ) -> void {
        encode(
            command_t::RESOLVE_SUBRESOURCE,
            resolve_subresource_t {
                pDstResource,
                DstSubresource,
                pSrcResource,
                SrcSubresource,
                Format,
            }
        );
    }

    virtual auto cmd_set_descriptor_heaps(
        UINT                        NumDescriptorHeaps,
        ID3D12DescriptorHeap *const *ppDescriptorHeaps
    ) -> void {
        set_descriptor_heaps_t cmd_data { NumDescriptorHeaps };
        for (auto i : range(NumDescriptorHeaps)) {
            cmd_data.DescriptorHeaps[i] = ppDescriptorHeaps[i];
        }
        encode(command_t::SET_DESCRIPTOR_HEAPS, cmd_data);
    }

    virtual auto cmd_set_compute_root_signature(ID3D12RootSignature *pRootSignature) -> void {
        encode(command_t::SET_COMPUTE_ROOT_SIGNATURE, set_compute_root_signature_t { pRootSignature });
    }

    virtual auto cmd_set_graphics_root_signature(ID3D12RootSignature *pRootSignature) -> void {
        encode(command_t::SET_GRAPHICS_ROOT_SIGNATURE, set_graphics_root_signature_t { pRootSignature });
    }

    virtual auto cmd_set_compute_root_descriptor_table(
        UINT                        RootParameterIndex,
        D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor
    ) -> void {
        encode(
            command_t::SET_COMPUTE_ROOT_DESCRIPTOR_TABLE,
            set_compute_root_descriptor_table_t {
                RootParameterIndex,
                BaseDescriptor,
            }
        );
    }

    virtual auto cmd_set_compute_root_constant(
        UINT RootParameterIndex,
        UINT SrcData,
        UINT DestOffsetIn32BitValues
    ) -> void {
        encode(
            command_t::SET_COMPUTE_ROOT_CONSTANT,
            set_compute_root_constant_t {
                RootParameterIndex,
                SrcData,
                DestOffsetIn32BitValues,
            }
        );
    }

    virtual auto cmd_set_compute_root_constants(
        UINT RootParameterIndex,
        UINT Num32BitValuesToSet,
        const void *pSrcData,
        UINT DestOffsetIn32BitValues
    ) -> void {
        this->commands.push_back(command_t::SET_COMPUTE_ROOT_CONSTANTS);
    }

    virtual auto cmd_set_compute_root_shader_resource_view(
        UINT                      RootParameterIndex,
        D3D12_GPU_VIRTUAL_ADDRESS BufferLocation
    ) -> void {
        encode(
            command_t::SET_COMPUTE_ROOT_SHADER_RESOURCE_VIEW,
            set_compute_root_shader_resource_view_t {
                RootParameterIndex,
                BufferLocation,
            }
        );
    }

    virtual auto cmd_set_graphics_root_constant(
        UINT RootParameterIndex,
        UINT SrcData,
        UINT DestOffsetIn32BitValues
    ) -> void {
        encode(
            command_t::SET_GRAPHICS_ROOT_CONSTANT,
            set_graphics_root_constant_t {
                RootParameterIndex,
                SrcData,
                DestOffsetIn32BitValues,
            }
        );
    }

    virtual auto cmd_set_graphics_root_descriptor_table(
        UINT                        RootParameterIndex,
        D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor
    ) -> void {
        encode(
            command_t::SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE,
            set_graphics_root_descriptor_table_t {
                RootParameterIndex,
                BaseDescriptor,
            }
        );
    }

    virtual auto cmd_clear_render_target_view(
        D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetView,
        const FLOAT                 ColorRGBA[4],
        UINT                        NumRects,
        const D3D12_RECT            *pRects
    ) -> void {
        this->commands.push_back(command_t::CLEAR_RENDER_TARGET_VIEW);
    }

    virtual auto cmd_clear_depth_stencil_view(
        D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView,
        D3D12_CLEAR_FLAGS           ClearFlags,
        FLOAT                       Depth,
        UINT8                       Stencil,
        UINT                        NumRects,
        const D3D12_RECT            *pRects
    ) -> void {
        this->commands.push_back(command_t::CLEAR_DEPTH_STENCIL_VIEW);
    }

    virtual auto cmd_set_render_targets(
        UINT                                NumRenderTargetDescriptors,
        const D3D12_CPU_DESCRIPTOR_HANDLE   *pRenderTargetDescriptors,
        BOOL                                RTsSingleHandleToDescriptorRange,
        const D3D12_CPU_DESCRIPTOR_HANDLE   *pDepthStencilDescriptor
    ) -> void {
        this->commands.push_back(command_t::SET_RENDER_TARGET);
    }

    virtual auto cmd_set_primitive_topolgy(D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopology) -> void {
        encode(
            command_t::SET_PRIMITIVE_TOPOLOGY,
            set_primitive_topology_t { PrimitiveTopology }
        );
    }

    virtual auto cmd_set_scissors(UINT NumRects, const D3D12_RECT *pRects) -> void {
        this->commands.push_back(command_t::SET_SCISSORS);
    }

    virtual auto cmd_set_viewports(UINT NumViewports, const D3D12_VIEWPORT *pViewports) -> void {
        this->commands.push_back(command_t::SET_VIEWPORTS);
    }

    virtual auto cmd_set_blend_factor(const FLOAT BlendFactor[4]) -> void {
        this->commands.push_back(command_t::SET_BLEND_FACTOR);
    }

    virtual auto cmd_set_stencil_ref(UINT StencilRef) -> void {
        encode(
            command_t::SET_STENCIL_REF,
            set_stencil_ref_t { StencilRef }
        );
    }

    virtual auto cmd_set_depth_bounds(FLOAT Min, FLOAT Max) -> void {
        encode(
            command_t::SET_DEPTH_BOUNDS,
            set_depth_bounds_t { Min, Max }
        );
    }

    virtual auto cmd_set_index_buffer(const D3D12_INDEX_BUFFER_VIEW *pView) -> void {
        this->commands.push_back(command_t::SET_INDEX_BUFFER);
    }

    virtual auto cmd_copy_buffer_region(
        ID3D12Resource *pDstBuffer,
        UINT64         DstOffset,
        ID3D12Resource *pSrcBuffer,
        UINT64         SrcOffset,
        UINT64         NumBytes
    ) -> void {
        this->commands.push_back(command_t::COPY_BUFFER_REGION);
    }

    virtual auto cmd_copy_texture_region(
        const D3D12_TEXTURE_COPY_LOCATION *pDst,
        UINT                        DstX,
        UINT                        DstY,
        UINT                        DstZ,
        const D3D12_TEXTURE_COPY_LOCATION *pSrc,
        const D3D12_BOX                   *pSrcBox
    ) -> void {
        this->commands.push_back(command_t::COPY_TEXTURE_REGION);
    }

    virtual auto cmd_set_pipeline_state(ID3D12PipelineState *pPipelineState) -> void {
        encode(
            command_t::SET_PIPELINE_STATE,
            set_pipeline_state_t { pPipelineState }
        );
    }

    virtual auto cmd_dispatch(
        UINT ThreadGroupCountX,
        UINT ThreadGroupCountY,
        UINT ThreadGroupCountZ
    ) -> void {
        encode(
            command_t::DISPATCH,
            dispatch_t {
                ThreadGroupCountX,
                ThreadGroupCountY,
                ThreadGroupCountZ,
            }
        );
    }

    virtual auto cmd_draw_instanced(
        UINT VertexCountPerInstance,
        UINT InstanceCount,
        UINT StartVertexLocation,
        UINT StartInstanceLocation
    ) -> void {
        encode(
            command_t::DRAW_INSTANCED,
            draw_instanced_t {
                VertexCountPerInstance,
                InstanceCount,
                StartVertexLocation,
                StartInstanceLocation,
            }
        );
    }

    virtual auto cmd_draw_indexed_instanced(
        UINT IndexCountPerInstance,
        UINT InstanceCount,
        UINT StartIndexLocation,
        INT  BaseVertexLocation,
        UINT StartInstanceLocation
    ) -> void {
        encode(
            command_t::DRAW_INDEXED_INSTANCED,
            draw_indexed_instanced_t {
                IndexCountPerInstance,
                InstanceCount,
                StartIndexLocation,
                BaseVertexLocation,
                StartInstanceLocation,
            }
        );
    }

    virtual auto cmd_execute_indirect(
        ID3D12CommandSignature *pCommandSignature,
        UINT                   MaxCommandCount,
        ID3D12Resource         *pArgumentBuffer,
        UINT64                 ArgumentBufferOffset,
        ID3D12Resource         *pCountBuffer,
        UINT64                 CountBufferOffset
    ) -> void {
        encode(
            command_t::EXECUTE_INDIRECT,
            execute_indirect_t {
                pCommandSignature,
                MaxCommandCount,
                pArgumentBuffer,
                ArgumentBufferOffset,
                pCountBuffer,
                CountBufferOffset,
            }
        );
    }

    virtual auto cmd_resource_barrier(
        UINT                   NumBarriers,
        const D3D12_RESOURCE_BARRIER *pBarriers
    ) -> void {
        this->commands.push_back(command_t::RESOURCE_BARRIER);
        // TODO
    }

    virtual auto cmd_set_vertex_buffers(
        UINT                     StartSlot,
        UINT                     NumViews,
        const D3D12_VERTEX_BUFFER_VIEW *pViews
    ) -> void {
        this->commands.push_back(command_t::SET_VERTEX_BUFFERS);
        encode_raw(StartSlot);
        encode_raw(NumViews);
        for (auto i : range(NumViews)) {
            encode_raw(pViews[i]);
        }
    }

private:
    template<typename T>
    auto encode(command_t cmd, T cmd_data) {
        this->commands.push_back(cmd);
        encode_raw(cmd_data);
    }

    template<typename T>
    auto encode_raw(T data) {
        auto const raw_data { reinterpret_cast<uint8_t *>(&data) };
        for (auto i : range(sizeof(T))) {
            this->data.push_back(raw_data[i]);
        }
    }

private:
    std::vector<command_t> commands;
    std::vector<uint8_t> data;
};
