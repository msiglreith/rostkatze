
#include "icd.hpp"

#include <vk_icd.h>
#include <cassert>
#include <algorithm>
#include <iostream>

extern "C" {

#undef VKAPI_ATTR
#define VKAPI_ATTR __declspec(dllexport)

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName) {
    return vkGetInstanceProcAddr(instance, pName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetPhysicalDeviceProcAddr(VkInstance instance, const char *pName) {
    VK_PHYSICAL_DEVICE_FNC()

    return nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t *pSupportedVersion) {
    *pSupportedVersion = std::min<uint32_t>(*pSupportedVersion, CURRENT_LOADER_ICD_INTERFACE_VERSION);
    return VK_SUCCESS;
}

}; // extern "C"