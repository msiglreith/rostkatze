#pragma once

#include "impl.hpp"
#include "device.hpp"

#include "command_recorder.hpp"

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
    command_buffer_t(
        ID3D12CommandAllocator* allocator,
        ComPtr<ID3D12GraphicsCommandList2> cmd_list,
        device_t* device,
        ID3D12DescriptorHeap* heap_gpu_cbv_srv_uav,
        ID3D12DescriptorHeap* heap_cpu_sampler
    ) :
        loader_magic { ICD_LOADER_MAGIC },
        allocator { allocator },
        command_list { cmd_list },
        command_recorder { cmd_list.Get() },
        heap_gpu_cbv_srv_uav { heap_gpu_cbv_srv_uav },
        heap_cpu_sampler { heap_cpu_sampler },
        heap_gpu_sampler { nullptr },
        pass_cache { std::nullopt },
        active_slot { std::nullopt },
        active_pipeline { nullptr },
        dynamic_state_dirty { false },
        viewports_dirty { false },
        scissors_dirty { false },
        num_viewports_scissors { 0 },
        vertex_buffer_views_dirty { 0 },
        _device { device },
        dynamic_state { },
        index_type { VK_INDEX_TYPE_UINT16 }
    { }

    ~command_buffer_t() {}

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

    auto bind_descriptor_heaps() {
        std::array<ID3D12DescriptorHeap *const, 2> heaps {
            this->heap_gpu_cbv_srv_uav,
            this->heap_gpu_sampler
        };
        this->command_recorder.cmd_set_descriptor_heaps(
            this->heap_gpu_sampler ? 2 : 1,
            &heaps[0]
        );
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

        const auto start_cbv_srv_uav { this->heap_gpu_cbv_srv_uav->GetGPUDescriptorHandleForHeapStart() };
        const auto start_cpu_sampler { this->heap_cpu_sampler->GetCPUDescriptorHandleForHeapStart() };
        auto start_sampler {
            this->heap_gpu_sampler ?
                this->heap_gpu_sampler->GetGPUDescriptorHandleForHeapStart() :
                D3D12_GPU_DESCRIPTOR_HANDLE { 0 }
        };

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

        std::bitset<D3D12_MAX_ROOT_COST> active_sampler_sets { 0 };
        for (auto i : range(num_table_entries)) {
            const auto data_slot { slot.pipeline->num_root_constants + i };
            if (!user_data.dirty[data_slot]) {
                continue;
            }

            if (user_data.type[data_slot] == pipeline_slot_t::data_entry_type::SAMPLER) {
                // Only cache samplers as they are currently living inside a CPU heap only.
                active_sampler_sets.set(i);
            }
        }

        if (active_sampler_sets.any()) {
            // TODO: thread safe access to descriptors
            auto num_sampler_sets { 0 };

            // Find a suitable gpu heap to bind.
            std::map<size_t, size_t> potential_heaps;
            for (auto i : range(num_table_entries)) {
                if (!active_sampler_sets[i]) {
                    continue;
                }

                auto const data_slot { slot.pipeline->num_root_constants + i };
                auto const offset { user_data.data[data_slot] };
                auto const handle { D3D12_CPU_DESCRIPTOR_HANDLE { start_cpu_sampler.ptr + offset } };

                auto const sampler_set { this->_device->descriptors_gpu_sampler.get_set(offset) };
                assert(sampler_set->set_sampler);
                num_sampler_sets += 1;

                auto const& placed_heaps { sampler_set->set_sampler->heaps_placed };
                for (auto const& heap : placed_heaps) {
                    auto ret { potential_heaps.insert(std::make_pair(heap, 1)) };
                    if (!ret.second) {
                        ret.first->second += 1;
                    }
                }
            }

            // Select heap from potential heaps
            auto heap { 0 };
            auto heap_it { std::max_element(potential_heaps.begin(), potential_heaps.end()) };
            if (heap_it == potential_heaps.end()) {
                // Haven't found any heap to use, so create a new one.
                heap = this->_device->descriptors_gpu_sampler.add_gpu_heap();
            } else if (num_sampler_sets == heap_it->second) {
                heap = heap_it->first;
            } else {
                assert(!"unimplemented");
            }
            auto& descriptor_heap { this->_device->descriptors_gpu_sampler.sampler_heap(heap) };
            if (this->heap_gpu_sampler != descriptor_heap.heap.heap()) {
                this->heap_gpu_sampler = descriptor_heap.heap.heap();
                start_sampler = this->heap_gpu_sampler->GetGPUDescriptorHandleForHeapStart();
                bind_descriptor_heaps();
            }

            // Upload descriptor sets into selected heap and set descriptor tables
            for (auto i : range(num_table_entries)) {
                if (!active_sampler_sets[i]) {
                    continue;
                }

                auto const data_slot { slot.pipeline->num_root_constants + i };
                auto const offset { user_data.data[data_slot] };
                auto const cpu_handle { D3D12_CPU_DESCRIPTOR_HANDLE { start_cpu_sampler.ptr + offset } };

                auto sampler_set { this->_device->descriptors_gpu_sampler.get_set(offset) };
                assert(sampler_set->set_sampler);

                auto& placed_heaps { sampler_set->set_sampler->heaps_placed };
                D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle { 0 };

                auto placed_heap { placed_heaps.begin() };
                if (placed_heap == placed_heaps.end()) {
                    // Not placed inside the heap so far
                    sampler_set->set_sampler->heaps_placed.emplace(heap);

                    // Allocate a new slice in the gpu heap
                    auto const num_descriptors { sampler_set->set_sampler->num_descriptors };
                    auto const sampler_handle { descriptor_heap.heap.alloc(num_descriptors) };
                    descriptor_heap.placed_sets.emplace(sampler_set, placed_descriptor_set_t { 1, sampler_handle, num_descriptors });
                    auto const [cpu_sampler, gpu_sampler] { sampler_handle };
                    gpu_handle = gpu_sampler;

                    // Upload cpu samplers into gpu heap
                    device()->CopyDescriptors(
                        1,
                        &cpu_sampler,
                        &num_descriptors,
                        1,
                        &cpu_handle,
                        &num_descriptors,
                        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
                    );
                } else {
                    auto& placed_set { descriptor_heap.placed_sets[sampler_set] };
                    placed_set.ref_count += 1; // TODO: sync
                    gpu_handle = std::get<1>(placed_set.start);
                }

                user_data.data[data_slot] = gpu_handle.ptr - start_sampler.ptr;
            }
        }

        for (auto i : range(num_table_entries)) {
            const auto data_slot { slot.pipeline->num_root_constants + i };
            if (!user_data.dirty[data_slot]) {
                continue;
            }

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

    auto bind_graphics_slot(draw_type draw_type) -> void {
        if (!this->active_slot) {
            bind_descriptor_heaps();
        }

        if (
            (this->dynamic_state_dirty && this->graphics_slot.pipeline->dynamic_states) ||
            this->active_slot != SLOT_GRAPHICS // Check if we are switching to Graphics
        ) {
            this->active_pipeline = nullptr;
        }

        if (!this->active_pipeline) {
            this->command_recorder.cmd_set_pipeline_state(
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
            this->command_recorder.cmd_set_viewports(
                this->num_viewports_scissors,
                this->viewports
            );
            this->viewports_dirty = false;
        }

        if (this->scissors_dirty) {
            this->command_recorder.cmd_set_scissors(
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
                    this->command_recorder.cmd_set_vertex_buffers(
                        static_cast<UINT>(*in_range),
                        static_cast<UINT>(i - *in_range),
                        this->vertex_buffer_views
                    );
                    in_range = std::nullopt;
                }
            }

            if (in_range) {
                this->command_recorder.cmd_set_vertex_buffers(
                    static_cast<UINT>(*in_range),
                    static_cast<UINT>(MAX_VERTEX_BUFFER_SLOTS - *in_range),
                    this->vertex_buffer_views
                );
            }
        }

        update_user_data(
            this->graphics_slot,
            [&] (UINT slot, uint32_t data, UINT offset) {
                this->command_recorder.cmd_set_graphics_root_constant(
                    slot,
                    data,
                    offset
                );
            },
            [&] (UINT slot, D3D12_GPU_DESCRIPTOR_HANDLE handle) {
                this->command_recorder.cmd_set_graphics_root_descriptor_table(
                    slot,
                    handle
                );
            }
        );
    }

    auto bind_compute_slot() -> void {
        if (!this->active_slot) {
            bind_descriptor_heaps();
        }

        if (this->active_slot != SLOT_COMPUTE) {
            this->active_pipeline = nullptr;
            this->active_slot = SLOT_COMPUTE;
        }

        if (!this->active_pipeline) {
            this->command_recorder.cmd_set_pipeline_state(
                std::get<pipeline_t::unique_pso_t>(this->compute_slot.pipeline->pso).pipeline.Get()
            );
        }

        update_user_data(
            this->compute_slot,
            [&] (UINT slot, uint32_t data, UINT offset) {
                this->command_recorder.cmd_set_compute_root_constant(
                    slot,
                    data,
                    offset
                );
            },
            [&] (UINT slot, D3D12_GPU_DESCRIPTOR_HANDLE handle) {
                this->command_recorder.cmd_set_compute_root_descriptor_table(
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
                    this->command_recorder.cmd_resource_barrier(1,
                        &CD3DX12_RESOURCE_BARRIER::Transition(view->image, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)
                    );
                    this->command_recorder.cmd_clear_render_target_view(
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
                    this->command_recorder.cmd_resource_barrier(1,
                        &CD3DX12_RESOURCE_BARRIER::Transition(view->image, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE)
                    );
                    this->command_recorder.cmd_clear_depth_stencil_view(
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

        this->command_recorder.cmd_set_render_targets(
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
                    this->command_recorder.cmd_resource_barrier(1,
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
                    this->command_recorder.cmd_resource_barrier(1,
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

            this->command_recorder.resolve_subresource(
                resolve_view->image,
                0, // TODO: D3D12CalcSubresource(resolve_view->)
                color_view->image,
                0, // TODO
                formats[render_pass->attachments[color_attachment.attachment].desc.format]
            );
        }
    }

    auto raw_command_list() const -> ID3D12CommandList* {
        return this->command_list.Get();
    }

private:
    /// Dispatchable object
    uintptr_t loader_magic;

    ComPtr<ID3D12GraphicsCommandList2> command_list;

public:
    command_buffer_recorder_native_t command_recorder;

    std::optional<pipeline_slot_type> active_slot;
    ID3D12PipelineState* active_pipeline;

    ID3D12DescriptorHeap* heap_gpu_cbv_srv_uav; // gpu
    ID3D12DescriptorHeap* heap_cpu_sampler; // cpu
    ID3D12DescriptorHeap* heap_gpu_sampler; // current gpu heap

    bool dynamic_state_dirty;
    bool viewports_dirty;
    bool scissors_dirty;
    std::bitset<MAX_VERTEX_BUFFER_SLOTS> vertex_buffer_views_dirty;

    // Currently set dynamic state
    dynamic_state_t dynamic_state;
    VkIndexType index_type;

    device_t* _device;

private:
    // Owning command allocator, required for reset
    ID3D12CommandAllocator* allocator;

public:
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
