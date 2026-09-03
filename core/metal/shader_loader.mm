#import "shader_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>
#include <filesystem>
#include <unistd.h>

namespace fs = std::filesystem;

namespace metal_llm {

static bool resolve_file_path(
    const std::string& include_name,
    const fs::path& current_dir,
    const std::vector<std::string>& search_paths,
    fs::path& resolved_path)
{
    // 1. Try relative to current file directory
    fs::path cand = current_dir / include_name;
    if (fs::exists(cand)) {
        resolved_path = fs::canonical(cand);
        return true;
    }

    // 2. Try current working directory
    cand = fs::current_path() / include_name;
    if (fs::exists(cand)) {
        resolved_path = fs::canonical(cand);
        return true;
    }

    // 3. Try each search path
    for (const auto& sp : search_paths) {
        cand = fs::path(sp) / include_name;
        if (fs::exists(cand)) {
            resolved_path = fs::canonical(cand);
            return true;
        }
    }

    return false;
}

static void expand_recursive(
    const fs::path& file_path,
    const std::vector<std::string>& search_paths,
    std::set<std::string>& visited_canonical_paths,
    std::ostringstream& out_stream)
{
    std::string can_str;
    try {
        can_str = fs::canonical(file_path).string();
    } catch (...) {
        can_str = file_path.string();
    }

    if (visited_canonical_paths.find(can_str) != visited_canonical_paths.end()) {
        return; // Already included via #pragma once
    }

    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "[ShaderLoader] Error opening file: " << file_path << std::endl;
        return;
    }

    fs::path current_dir = file_path.parent_path();
    std::string line;
    bool has_pragma_once = false;

    while (std::getline(file, line)) {
        // Strip leading whitespace
        size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos) {
            out_stream << "\n";
            continue;
        }

        std::string trimmed = line.substr(first);

        if (trimmed == "#pragma once") {
            has_pragma_once = true;
            visited_canonical_paths.insert(can_str);
            continue;
        }

        if (trimmed.rfind("#include \"", 0) == 0) {
            size_t end_quote = trimmed.find('"', 10);
            if (end_quote != std::string::npos) {
                std::string inc_file = trimmed.substr(10, end_quote - 10);
                fs::path resolved;
                if (resolve_file_path(inc_file, current_dir, search_paths, resolved)) {
                    expand_recursive(resolved, search_paths, visited_canonical_paths, out_stream);
                } else {
                    std::cerr << "[ShaderLoader] Could not resolve #include \"" << inc_file << "\" from " << file_path << std::endl;
                }
                continue;
            }
        }

        out_stream << line << "\n";
    }

    if (has_pragma_once) {
        visited_canonical_paths.insert(can_str);
    }
}

std::string expand_shader_source(
    const std::string& entry_file_path,
    const std::vector<std::string>& search_paths)
{
    std::set<std::string> visited;
    std::ostringstream ss;
    fs::path entry(entry_file_path);
    if (!fs::exists(entry)) {
        std::cerr << "[ShaderLoader] Entry file does not exist: " << entry_file_path << std::endl;
        return "";
    }
    expand_recursive(fs::canonical(entry), search_paths, visited, ss);
    return ss.str();
}

id<MTLLibrary> load_modular_metal_library(
    id<MTLDevice> device,
    const std::string& entry_file_path,
    const std::vector<std::string>& search_paths)
{
    std::string expanded = expand_shader_source(entry_file_path, search_paths);
    if (expanded.empty()) {
        std::cerr << "[ShaderLoader] Empty shader source for: " << entry_file_path << std::endl;
        return nil;
    }

    NSString* ns_src = [NSString stringWithUTF8String:expanded.c_str()];
    NSError* error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:ns_src options:nil error:&error];
    if (!lib || error) {
        std::cerr << "[ShaderLoader] Metal compilation error for " << entry_file_path << ":\n"
                  << (error ? [[error localizedDescription] UTF8String] : "Unknown error")
                  << std::endl;
        return nil;
    }
    return lib;
}

} // namespace metal_llm
