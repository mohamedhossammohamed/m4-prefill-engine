#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string>
#include <vector>

namespace metal_llm {

// Recursively resolves and expands all `#include "..."` statements in an MSL shader source file,
// honoring `#pragma once` inclusion guards, and compiles it into an `id<MTLLibrary>`.
id<MTLLibrary> load_modular_metal_library(
    id<MTLDevice> device,
    const std::string& entry_file_path,
    const std::vector<std::string>& search_paths = {});

// Expands shader source into a single self-contained MSL string
std::string expand_shader_source(
    const std::string& entry_file_path,
    const std::vector<std::string>& search_paths = {});

} // namespace metal_llm
