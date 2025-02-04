use cmake::Config;
use globwalk::{FileType, GlobWalkerBuilder};
use std::env;
use std::path::PathBuf;

fn get_lib_dir() -> PathBuf {
    let base_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    base_dir.join("lib")
}

fn get_built_libs() -> Vec<String> {
    let lib_dir = get_lib_dir();
    let walker =
        GlobWalkerBuilder::from_patterns(lib_dir.to_str().unwrap(), &["*.{lib,dylib,a,so}"])
            .file_type(FileType::FILE)
            .max_depth(4)
            .follow_links(true)
            .build()
            .unwrap();
    walker
        .into_iter()
        .filter_map(Result::ok)
        .map(|p| {
            let stem = p.path().file_stem().unwrap().to_str().unwrap();
            let lib_name = stem.strip_prefix("lib").unwrap_or(stem);
            //if cfg!(debug_assertions) {
            //    println!("cargo:warning=[DEBUG] Found lib: {lib_name}");
            //}
            lib_name.to_string()
        })
        .collect::<Vec<String>>()
}

fn main() {
    // Build the C++ project using CMake
    let mut conf = Config::new("llama.cpp");
    conf
        //.profile("Release")
        //.build_target("ggml")
        //.build_target("llama")
        //.very_verbose(true)
        .define("CMAKE_CXX_STANDARD", "17")
        .define("CMAKE_CXX_STANDARD_REQUIRED", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF");

    //if cfg!(debug_assertions) {
    //    conf.very_verbose(true);
    //}

    // Llama features
    if cfg!(target_os = "macos") {
        conf.define("GGML_BLAS", "OFF");
        conf.define("GGML_OPENMP", "OFF");
    }

    if cfg!(feature = "cuda") {
        conf.define("GGML_CUDA", "ON");
    } else {
        conf.define("GGML_CUDA", "OFF");
    }

    if cfg!(feature = "vulkan") && cfg!(target_os = "linux") {
        conf.define("GGML_VULKAN", "ON");
        println!("cargo:rustc-link-lib=vulkan");
    } else {
        conf.define("GGML_VULKAN", "OFF");
    }

    // Linking
    let lib_type = if cfg!(feature = "sharedlib") {
        conf.define("BUILD_SHARED_LIBS", "ON");
        "dylib"
    } else {
        conf.define("BUILD_SHARED_LIBS", "OFF");
        //conf.define("GGML_STATIC", "ON");
        "static"
    };

    let dst = conf.build();

    // Tell cargo where to find the built library
    // Add all possible library search paths
    println!(
        "cargo:rustc-link-search=native={}",
        get_lib_dir().to_str().unwrap()
    );
    // Link the libraries
    get_built_libs().into_iter().for_each(|l| {
        println!("cargo:rustc-link-lib={lib_type}={l}");
    });
    // // macOS
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    }

    // Tell cargo to rebuild if these files change
    println!("cargo:rerun-if-changed=llama.cpp");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.hpp");

    // Generate bindings
    let bindings = bindgen::builder();
    let binding_gens = bindings
        .header("wrapper.hpp")
        // Add include path from CMake build
        .clang_arg(format!("-I{}", dst.join("include").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("libllama_.*")
        .allowlist_type("libllama_.*")
        .prepend_enum_name(false)
        .generate()
        .expect("Unable to generate bindings");

    // Write to src
    let src_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    binding_gens
        .write_to_file(src_dir.join("src").join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Write to out dir
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    binding_gens
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");
}
