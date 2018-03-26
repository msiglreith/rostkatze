cmake_minimum_required(VERSION 2.8)

project(rostkatze)

add_compile_options("/std:c++latest")

include_directories(include)
include_directories(deps)
include_directories(deps/stdx/external/fmt)
include_directories(deps/stdx/include)

## dependencies
# stdx
add_subdirectory(deps/stdx)
# gsl
include_directories(deps/gsl/include)
# vulkan
include_directories(deps/vulkan)
# spdlog
include_directories(deps/spdlog/include)
# SPIRV-Cross
add_subdirectory(deps/spirv-cross)
# gsl
include_directories(deps/D3DX12)

add_library(rostkatze SHARED
    src/command_list.cpp
    src/formats.cpp
    src/lib.cpp
    src/impl.cpp
    src/wsi.cpp
)

target_link_libraries(rostkatze dxgi d3d12 d3dcompiler spirv-cross-hlsl)
