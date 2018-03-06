#pragma once

#define NOMINMAX

#include <DXGIFormat.h>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan.hpp>

#include <array>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/msvc_sink.h>

static auto init_logging() {
    static auto initialized { false };
    if (initialized) {
        return;
    }

    initialized = true;
    // spdlog::basic_logger_mt("rostkatze", "test.log");
    spdlog::create<spdlog::sinks::msvc_sink_mt>("rostkatze");
    spdlog::set_level(spdlog::level::debug);
}

static auto log() {
    return spdlog::get("rostkatze");
}

#define TRACE(...) log()->trace(__VA_ARGS__)
#define DEBUG(...) log()->debug(__VA_ARGS__)
#define INFO(...)  log()->info(__VA_ARGS__)
#define WARN(...)  log()->warn(__VA_ARGS__)
#define ERR(...)   log()->error(__VA_ARGS__)

struct format_block_t {
    uint8_t width;
    uint8_t height;
    uint16_t bits;
};

extern std::array<DXGI_FORMAT, VK_FORMAT_RANGE_SIZE> formats;
extern std::array<format_block_t, VK_FORMAT_RANGE_SIZE> formats_block;
extern std::array<VkFormatProperties, VK_FORMAT_RANGE_SIZE> formats_property;

#define VK_FNC(name) \
    if (!strcmp(pName, #name)) { \
        return reinterpret_cast<PFN_vkVoidFunction>(&name); \
    }

#define VK_INSTANCE_FNC() \
    VK_FNC(vkCreateInstance) \
    VK_FNC(vkDestroyInstance) \
    VK_FNC(vkEnumerateInstanceExtensionProperties) \
    VK_FNC(vkEnumerateDeviceExtensionProperties) \
    VK_FNC(vkGetDeviceProcAddr) \
    VK_FNC(vkEnumeratePhysicalDevices) \
    VK_FNC(vkCreateDevice) \
    VK_FNC(vkCreateWin32SurfaceKHR)

#define VK_PHYSICAL_DEVICE_FNC() \
    VK_FNC(vkGetPhysicalDeviceFeatures) \
    VK_FNC(vkGetPhysicalDeviceFormatProperties) \
    VK_FNC(vkGetPhysicalDeviceImageFormatProperties) \
    VK_FNC(vkGetPhysicalDeviceProperties) \
    VK_FNC(vkGetPhysicalDeviceMemoryProperties) \
    VK_FNC(vkGetPhysicalDeviceQueueFamilyProperties) \
    VK_FNC(vkGetPhysicalDeviceSparseImageFormatProperties) \
    VK_FNC(vkGetPhysicalDeviceSurfaceSupportKHR) \
    VK_FNC(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) \
    VK_FNC(vkGetPhysicalDeviceSurfaceFormatsKHR) \
    VK_FNC(vkGetPhysicalDeviceSurfacePresentModesKHR) \
    VK_FNC(vkGetPhysicalDeviceFeatures2KHR) \
    VK_FNC(vkGetPhysicalDeviceProperties2KHR) \
    VK_FNC(vkGetPhysicalDeviceFormatProperties2KHR) \
    VK_FNC(vkGetPhysicalDeviceImageFormatProperties2KHR) \
    VK_FNC(vkGetPhysicalDeviceQueueFamilyProperties2KHR) \
    VK_FNC(vkGetPhysicalDeviceMemoryProperties2KHR) \
    VK_FNC(vkGetPhysicalDeviceSparseImageFormatProperties2KHR)

#define VK_DEVICE_FNC() \
    VK_FNC(vkDestroyDevice) \
    VK_FNC(vkGetDeviceQueue) \
    VK_FNC(vkQueueSubmit) \
    VK_FNC(vkQueueWaitIdle) \
    VK_FNC(vkDeviceWaitIdle) \
    VK_FNC(vkAllocateMemory) \
    VK_FNC(vkFreeMemory) \
    VK_FNC(vkMapMemory) \
    VK_FNC(vkUnmapMemory) \
    VK_FNC(vkFlushMappedMemoryRanges) \
    VK_FNC(vkInvalidateMappedMemoryRanges) \
    VK_FNC(vkGetDeviceMemoryCommitment) \
    VK_FNC(vkBindBufferMemory) \
    VK_FNC(vkBindImageMemory) \
    VK_FNC(vkGetBufferMemoryRequirements) \
    VK_FNC(vkGetImageMemoryRequirements) \
    VK_FNC(vkGetImageSparseMemoryRequirements) \
    VK_FNC(vkQueueBindSparse) \
    VK_FNC(vkCreateFence) \
    VK_FNC(vkDestroyFence) \
    VK_FNC(vkResetFences) \
    VK_FNC(vkGetFenceStatus) \
    VK_FNC(vkWaitForFences) \
    VK_FNC(vkCreateSemaphore) \
    VK_FNC(vkDestroySemaphore) \
    VK_FNC(vkCreateEvent) \
    VK_FNC(vkDestroyEvent) \
    VK_FNC(vkGetEventStatus) \
    VK_FNC(vkSetEvent) \
    VK_FNC(vkResetEvent) \
    VK_FNC(vkCreateQueryPool) \
    VK_FNC(vkDestroyQueryPool) \
    VK_FNC(vkGetQueryPoolResults) \
    VK_FNC(vkCreateBuffer) \
    VK_FNC(vkDestroyBuffer) \
    VK_FNC(vkCreateBufferView) \
    VK_FNC(vkDestroyBufferView) \
    VK_FNC(vkCreateImage) \
    VK_FNC(vkDestroyImage) \
    VK_FNC(vkGetImageSubresourceLayout) \
    VK_FNC(vkCreateImageView) \
    VK_FNC(vkDestroyImageView) \
    VK_FNC(vkCreateShaderModule) \
    VK_FNC(vkDestroyShaderModule) \
    VK_FNC(vkCreatePipelineCache) \
    VK_FNC(vkDestroyPipelineCache) \
    VK_FNC(vkGetPipelineCacheData) \
    VK_FNC(vkMergePipelineCaches) \
    VK_FNC(vkCreateGraphicsPipelines) \
    VK_FNC(vkCreateComputePipelines) \
    VK_FNC(vkDestroyPipeline) \
    VK_FNC(vkCreatePipelineLayout) \
    VK_FNC(vkDestroyPipelineLayout) \
    VK_FNC(vkCreateSampler) \
    VK_FNC(vkDestroySampler) \
    VK_FNC(vkCreateDescriptorSetLayout) \
    VK_FNC(vkDestroyDescriptorSetLayout) \
    VK_FNC(vkCreateDescriptorPool) \
    VK_FNC(vkDestroyDescriptorPool) \
    VK_FNC(vkResetDescriptorPool) \
    VK_FNC(vkAllocateDescriptorSets) \
    VK_FNC(vkFreeDescriptorSets) \
    VK_FNC(vkUpdateDescriptorSets) \
    VK_FNC(vkCreateFramebuffer) \
    VK_FNC(vkDestroyFramebuffer) \
    VK_FNC(vkCreateRenderPass) \
    VK_FNC(vkDestroyRenderPass) \
    VK_FNC(vkGetRenderAreaGranularity) \
    VK_FNC(vkCreateCommandPool) \
    VK_FNC(vkDestroyCommandPool) \
    VK_FNC(vkResetCommandPool) \
    VK_FNC(vkAllocateCommandBuffers) \
    VK_FNC(vkFreeCommandBuffers) \
    VK_FNC(vkCreateSwapchainKHR) \
    VK_FNC(vkDestroySwapchainKHR) \
    VK_FNC(vkGetSwapchainImagesKHR) \
    VK_FNC(vkAcquireNextImageKHR) \
    VK_FNC(vkQueuePresentKHR) \
    VK_FNC(vkBeginCommandBuffer) \
    VK_FNC(vkEndCommandBuffer) \
    VK_FNC(vkResetCommandBuffer) \
    VK_FNC(vkCmdBindPipeline) \
    VK_FNC(vkCmdSetViewport) \
    VK_FNC(vkCmdSetScissor) \
    VK_FNC(vkCmdSetLineWidth) \
    VK_FNC(vkCmdSetDepthBias) \
    VK_FNC(vkCmdSetBlendConstants) \
    VK_FNC(vkCmdSetDepthBounds) \
    VK_FNC(vkCmdSetStencilCompareMask) \
    VK_FNC(vkCmdSetStencilWriteMask) \
    VK_FNC(vkCmdSetStencilReference) \
    VK_FNC(vkCmdBindDescriptorSets) \
    VK_FNC(vkCmdBindIndexBuffer) \
    VK_FNC(vkCmdBindVertexBuffers) \
    VK_FNC(vkCmdDraw) \
    VK_FNC(vkCmdDrawIndexed) \
    VK_FNC(vkCmdDrawIndirect) \
    VK_FNC(vkCmdDrawIndexedIndirect) \
    VK_FNC(vkCmdDispatch) \
    VK_FNC(vkCmdDispatchIndirect) \
    VK_FNC(vkCmdCopyBuffer) \
    VK_FNC(vkCmdCopyImage) \
    VK_FNC(vkCmdBlitImage) \
    VK_FNC(vkCmdCopyBufferToImage) \
    VK_FNC(vkCmdCopyImageToBuffer) \
    VK_FNC(vkCmdUpdateBuffer) \
    VK_FNC(vkCmdFillBuffer) \
    VK_FNC(vkCmdClearColorImage) \
    VK_FNC(vkCmdClearDepthStencilImage) \
    VK_FNC(vkCmdClearAttachments) \
    VK_FNC(vkCmdResolveImage) \
    VK_FNC(vkCmdSetEvent) \
    VK_FNC(vkCmdResetEvent) \
    VK_FNC(vkCmdWaitEvents) \
    VK_FNC(vkCmdPipelineBarrier) \
    VK_FNC(vkCmdBeginQuery) \
    VK_FNC(vkCmdEndQuery) \
    VK_FNC(vkCmdResetQueryPool) \
    VK_FNC(vkCmdWriteTimestamp) \
    VK_FNC(vkCmdCopyQueryPoolResults) \
    VK_FNC(vkCmdPushConstants) \
    VK_FNC(vkCmdBeginRenderPass) \
    VK_FNC(vkCmdNextSubpass) \
    VK_FNC(vkCmdEndRenderPass) \
    VK_FNC(vkCmdExecuteCommands)
