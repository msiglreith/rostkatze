
# Rostkatze

An offspring of the gfx-rs project, implementing a C++ Vulkan driver on top of D3D12.
Currently, lacks a lot functionality but can run quite a few Vulkan samples already.

## Missing pieces

- Pipeline barriers
- RenderPasses
- Secondary command buffers
- Simultaneous command buffer submission
- Queries
- Validation of the current implementation(!)
- Resource Tier 1 support

... (100+ bullet points)

## Running

To able to use it latest Windows 10 Update is required (atm) and a GPU with Resource Tier 2 hardware (no NVIDIA!).

The library can be built with VS 2017 (C++17 support). In order to use the `rostkatze` ICD set the `VK_ICD_FILENAMES` variable to the path of `rostkatze_debug.json`.

Good luck!


## "Working" samples (SaschaWillems)

- triangle
- texturearray
- mesh
- vulkanscene
- gears
- pipelines
- specialization constants
- pbrbasic
- pbribl
- pbrtexture
- scenerendering
- shadowmappingcascade
- sphericalenvmapping
- imgui
- shadowmappingomni
- radialblur
- bloom
- computeshader
- distancefieldrendering
- offscreen

## Partially working

- texture (inverse)
- shadowmapping (dynamic depth bias missing)
- computecloth (bugs)

# Almost working

- pushconstants (push constant array members)
- skeletalanimation (inverse)
- ssao (inverse, spec constants in array, ?)
- instancing (inverse, ?)

## SPIRV-Cross issues

- `inverse`
- Missing decorations for struct member fields
- spec constants in arrays
- pointcoord builtin not supported (even with compat mode)
