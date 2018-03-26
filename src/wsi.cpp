//! Windows system interface related implementations

#include "device.hpp"
#include "impl.hpp"

#include <stdx/match.hpp>

#include <Corewindow.h>

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

    const auto hr {
        std::visit(
            stdx::match(
                [&] (HWND hwnd) {
                    return surface->dxgi_factory->CreateSwapChainForHwnd(
                        device->present_queue.queue.Get(),
                        hwnd,
                        &desc,
                        nullptr, // TODO: fullscreen
                        nullptr, // TODO: restrict?
                        &dxgi_swapchain
                    );
                },
                [&] (IUnknown* window) {
                    return surface->dxgi_factory->CreateSwapChainForCoreWindow(
                        device->present_queue.queue.Get(),
                        window,
                        &desc,
                        nullptr, // TODO: restrict?
                        &dxgi_swapchain
                    );
                }
            ),
            surface->handle
        )
    };

    if (FAILED(hr)) {
        assert(false);
    }
    // TODO: errror

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

    const auto hwnd {
        std::visit(
            stdx::match(
                [&] (HWND hwnd) {
                    return hwnd;
                },
                [&] (IUnknown *window) {
                    ComPtr<ICoreWindowInterop> core_window { nullptr };
                    window->QueryInterface(core_window.GetAddressOf());

                    HWND hwnd;
                    core_window->get_WindowHandle(&hwnd);
                    return hwnd;
                }
            ),
            surface->handle
        )
    };

    RECT rect;
    if (!::GetClientRect(hwnd, &rect)) {
        // TODO
        ERR("Couldn't get size of window");
    }

    VkExtent2D extent {
        static_cast<uint32_t>(rect.right - rect.left),
        static_cast<uint32_t>(rect.bottom - rect.top),
    };

    *pSurfaceCapabilities = {
        // Image count due to FLIP_DISCARD
        2, // minImageCount
        16, // maxImageCount
        extent, // currentExtent
        extent, // minImageExtent
        extent, // maxImageExtent
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

    auto instance { reinterpret_cast<instance_t *>(_instance) };
    auto const& info { *pCreateInfo };

    *pSurface = reinterpret_cast<VkSurfaceKHR>(
        new surface_t { instance->dxgi_factory, info.hwnd}
    );

    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateUWPSurfaceRKZ(
    VkInstance                                  _instance,
    const VkUWPSurfaceCreateInfoRKZ*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface
) {
    TRACE("vkCreateUWPSurfaceRKZ");

    auto instance { reinterpret_cast<instance_t *>(_instance) };
    auto const& info { *pCreateInfo };

    *pSurface = reinterpret_cast<VkSurfaceKHR>(
        new surface_t { instance->dxgi_factory, info.pWindow}
    );

    return VK_SUCCESS;
}

VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceUWPPresentationSupportRKZ(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex
) {
    return queueFamilyIndex == QUEUE_FAMILY_GENERAL_PRESENT;
}
