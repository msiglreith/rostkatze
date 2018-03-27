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