cmake_minimum_required(VERSION 3.15...3.27)

# Use Clang 20 for C++23 support
set(CMAKE_C_COMPILER clang-20 CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER clang++-20 CACHE FILEPATH "C++ compiler" FORCE)

project(matrices_evolved LANGUAGES CXX)

# Build implementation selection
option(BUILD_RUST "Build Rust PyO3 implementation" ON)
option(BUILD_CPP "Build C++ nanobind implementation" ON)

# Force clean build on macro changes
set(MACRO_HASH "aligned_enabled")
set(HASH_FILE "${CMAKE_BINARY_DIR}/macro_hash.txt")
if(EXISTS "${HASH_FILE}")
    file(READ "${HASH_FILE}" OLD_HASH)
    if(NOT "${OLD_HASH}" STREQUAL "${MACRO_HASH}")
        message(STATUS "Macro configuration changed, cleaning build")
        file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/CMakeFiles/_event_signing_impl.dir")
    endif()
endif()
file(WRITE "${HASH_FILE}" "${MACRO_HASH}")

# Compiler information
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")

if (NOT SKBUILD)
  message(WARNING "This CMake file is meant to be executed using 'scikit-build-core'. Running it directly will almost certainly not produce the desired result. Use: pip install .")
endif()

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED PATHS ${nanobind_ROOT})

# Download and build dependencies
include(ExternalProject)

# Boost.JSON 
# its header only but this is kept for compatibility with the existing CMake structure
ExternalProject_Add(
  boost
  URL https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.gz
  CONFIGURE_COMMAND ./bootstrap.sh --with-libraries=json --with-toolset=clang-20
  BUILD_COMMAND rm project-config.jam && ./b2 toolset=clang-20 --with-json variant=release link=static cxxflags=-fPIC -O3 -static
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS <SOURCE_DIR>/stage/lib/libboost_json.a
  INSTALL_COMMAND ""
)

# SIMDE for portable SIMD
ExternalProject_Add(
  simde
  GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

# AWS-LC for cryptography
ExternalProject_Add(
  aws-lc
  GIT_REPOSITORY https://github.com/aws/aws-lc.git
  GIT_TAG main
  GIT_SHALLOW TRUE
  CMAKE_GENERATOR "Unix Makefiles"
  CMAKE_ARGS 
    -DCMAKE_BUILD_TYPE=Release 
    -DBUILD_SHARED_LIBS=OFF 
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_C_COMPILER=clang-20
    -DCMAKE_CXX_COMPILER=clang++-20
    -DCMAKE_C_FLAGS=-O3
    -DCMAKE_CXX_FLAGS=-O3
    -DDISABLE_PERL=ON
    -DDISABLE_GO=ON
    -DBUILD_TESTING=OFF
  BUILD_BYPRODUCTS <BINARY_DIR>/crypto/libcrypto.a
  INSTALL_COMMAND ""
)

ExternalProject_Get_Property(boost SOURCE_DIR)
set(BOOST_ROOT ${SOURCE_DIR})
ExternalProject_Get_Property(simde SOURCE_DIR)
set(SIMDE_ROOT ${SOURCE_DIR})
ExternalProject_Get_Property(aws-lc SOURCE_DIR BINARY_DIR)
set(AWS_LC_ROOT ${SOURCE_DIR})
set(AWS_LC_BUILD ${BINARY_DIR})

# AWS-LC library paths
message(STATUS "AWS-LC source: ${AWS_LC_ROOT}")
message(STATUS "AWS-LC build: ${AWS_LC_BUILD}")
message(STATUS "AWS-LC crypto lib: ${AWS_LC_BUILD}/crypto/libcrypto.a")

# Source files for the C++ implementation
set(CPP_SOURCES
    cpp/lib.cpp
    cpp/json/canonicalization.cpp
    cpp/crypto/ed25519.cpp
)

# Add all .cpp files from encoders and decoders
file(GLOB_RECURSE ENCODER_SOURCES "cpp/base64/encoders/*.cpp")
file(GLOB_RECURSE DECODER_SOURCES "cpp/base64/decoders/*.cpp")
list(APPEND CPP_SOURCES ${ENCODER_SOURCES} ${DECODER_SOURCES})

# Add header dependencies to force recompilation when headers change
set(CPP_HEADERS
    cpp/global.h
    cpp/json/canonicalization.h
    cpp/crypto/ed25519.h
)

# Add all .h files from encoders and decoders
file(GLOB_RECURSE ENCODER_HEADERS "cpp/base64/encoders/*.h")
file(GLOB_RECURSE DECODER_HEADERS "cpp/base64/decoders/*.h")
list(APPEND CPP_HEADERS ${ENCODER_HEADERS} ${DECODER_HEADERS})

nanobind_add_module(_event_signing_impl STABLE_ABI NOMINSIZE LTO ${CPP_SOURCES} ${CPP_HEADERS})
set_target_properties(_event_signing_impl PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN FALSE
)
add_dependencies(_event_signing_impl boost simde aws-lc)

target_include_directories(_event_signing_impl PRIVATE ${BOOST_ROOT} ${SIMDE_ROOT} ${AWS_LC_ROOT}/include)

# Create imported targets for external libraries
add_library(aws_lc_crypto STATIC IMPORTED)
set_target_properties(aws_lc_crypto PROPERTIES IMPORTED_LOCATION ${AWS_LC_BUILD}/crypto/libcrypto.a)
add_dependencies(aws_lc_crypto aws-lc)
# kept for compatibility with existing CMake structure or if we add something in future
add_library(boost_json STATIC IMPORTED)
set_target_properties(boost_json PROPERTIES IMPORTED_LOCATION ${BOOST_ROOT}/stage/lib/libboost_json.a)
add_dependencies(boost_json boost)

target_link_libraries(_event_signing_impl PRIVATE aws_lc_crypto boost_json)
target_link_options(_event_signing_impl PRIVATE -fuse-ld=lld-20)
set_target_properties(_event_signing_impl PROPERTIES LINK_FLAGS "-fuse-ld=lld-20")
target_compile_options(_event_signing_impl PRIVATE -fPIC)
target_compile_features(_event_signing_impl PRIVATE cxx_std_23)

# Check for C++23 feature availability
include(CheckIncludeFileCXX)
check_include_file_cxx("format" HAS_FORMAT_HEADER)
check_include_file_cxx("print" HAS_PRINT_HEADER)
check_include_file_cxx("flat_map" HAS_FLAT_MAP_HEADER)
check_include_file_cxx("mdspan" HAS_MDSPAN_HEADER)
check_include_file_cxx("generator" HAS_GENERATOR_HEADER)

if(HAS_FORMAT_HEADER)
    target_compile_definitions(_event_signing_impl PRIVATE HAS_FORMAT_HEADER=1)
    message(STATUS "C++23 <format> available")
endif()

if(HAS_PRINT_HEADER)
    target_compile_definitions(_event_signing_impl PRIVATE HAS_PRINT_HEADER=1)
    message(STATUS "C++23 <print> available")
endif()

if(HAS_FLAT_MAP_HEADER)
    target_compile_definitions(_event_signing_impl PRIVATE HAS_FLAT_MAP_HEADER=1)
    message(STATUS "C++23 <flat_map> available")
endif()

if(HAS_MDSPAN_HEADER)
    target_compile_definitions(_event_signing_impl PRIVATE HAS_MDSPAN_HEADER=1)
    message(STATUS "C++23 <mdspan> available")
endif()

if(HAS_GENERATOR_HEADER)
    target_compile_definitions(_event_signing_impl PRIVATE HAS_GENERATOR_HEADER=1)
    message(STATUS "C++23 <generator> available")
endif()

# Apply performance optimizations
function(apply_optimizations target)
    message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
    
    # Base optimization flags 
    target_compile_options(${target} PRIVATE
        -O3 -ffast-math -march=native -mtune=native
        -funroll-loops -fomit-frame-pointer
        -fno-stack-protector
        -funsafe-math-optimizations
        -fno-plt -fno-semantic-interposition
        -fmerge-all-constants
        -DNDEBUG -DBOOST_DISABLE_ASSERTS
    )
    
    include(CheckCXXCompilerFlag)
    
    # SIMD instruction sets
    check_cxx_compiler_flag("-mavx2" SUPPORTS_AVX2)
    if(SUPPORTS_AVX2)
        target_compile_options(${target} PRIVATE -mavx2)
        message(STATUS "AVX2 support enabled")
    endif()
    
    check_cxx_compiler_flag("-mfma" SUPPORTS_FMA)
    if(SUPPORTS_FMA)
        target_compile_options(${target} PRIVATE -mfma)
        message(STATUS "FMA support enabled")
    endif()
    
    check_cxx_compiler_flag("-mbmi2" SUPPORTS_BMI2)
    if(SUPPORTS_BMI2)
        target_compile_options(${target} PRIVATE -mbmi2)
        message(STATUS "BMI2 support enabled")
    endif()
    
    # Memory and cache optimizations
    check_cxx_compiler_flag("-fprefetch-loop-arrays" SUPPORTS_PREFETCH)
    if(SUPPORTS_PREFETCH)
        target_compile_options(${target} PRIVATE -fprefetch-loop-arrays)
        message(STATUS "Loop array prefetching enabled")
    endif()
    
    check_cxx_compiler_flag("-falign-functions=32" SUPPORTS_ALIGN_FUNC)
    if(SUPPORTS_ALIGN_FUNC)
        target_compile_options(${target} PRIVATE -falign-functions=32)
        message(STATUS "Function alignment optimization enabled")
    endif()
    
    check_cxx_compiler_flag("-falign-loops=32" SUPPORTS_ALIGN_LOOPS)
    if(SUPPORTS_ALIGN_LOOPS)
        target_compile_options(${target} PRIVATE -falign-loops=32)
        message(STATUS "Loop alignment optimization enabled")
    endif()
    
    # Compiler-specific optimizations
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "Applying Clang optimizations")
        
        # Clang-specific flags
        check_cxx_compiler_flag("-flto=thin" CLANG_SUPPORTS_THIN_LTO)
        if(CLANG_SUPPORTS_THIN_LTO)
            target_compile_options(${target} PRIVATE -flto=thin)
        endif()
        
        check_cxx_compiler_flag("-finline-functions" CLANG_SUPPORTS_INLINE_FUNCTIONS)
        if(CLANG_SUPPORTS_INLINE_FUNCTIONS)
            target_compile_options(${target} PRIVATE -finline-functions)
        endif()
        
        check_cxx_compiler_flag("-fvectorize" CLANG_SUPPORTS_VECTORIZE)
        if(CLANG_SUPPORTS_VECTORIZE)
            target_compile_options(${target} PRIVATE -fvectorize -fslp-vectorize)
            message(STATUS "Clang vectorization enabled")
        endif()
        
        # Advanced Clang features
        check_cxx_compiler_flag("-fforce-emit-vtables" CLANG_SUPPORTS_VTABLES)
        if(CLANG_SUPPORTS_VTABLES)
            target_compile_options(${target} PRIVATE -fforce-emit-vtables)
        endif()
        
        check_cxx_compiler_flag("-fstrict-vtable-pointers" CLANG_SUPPORTS_STRICT_VTABLES)
        if(CLANG_SUPPORTS_STRICT_VTABLES)
            target_compile_options(${target} PRIVATE -fstrict-vtable-pointers)
        endif()
        
        check_cxx_compiler_flag("-fwhole-program-vtables" CLANG_SUPPORTS_WHOLE_VTABLES)
        if(CLANG_SUPPORTS_WHOLE_VTABLES)
            target_compile_options(${target} PRIVATE -fwhole-program-vtables)
        endif()
        
        # LLVM-specific optimizations
        check_cxx_compiler_flag("-mllvm -polly" CLANG_SUPPORTS_POLLY)
        if(CLANG_SUPPORTS_POLLY)
            target_compile_options(${target} PRIVATE -mllvm -polly)
            message(STATUS "Polly loop optimization enabled")
        endif()
        
        # Profile-guided optimization
        check_cxx_compiler_flag("-fprofile-sample-accurate" CLANG_SUPPORTS_PROFILE_ACCURATE)
        if(CLANG_SUPPORTS_PROFILE_ACCURATE)
            target_compile_options(${target} PRIVATE -fprofile-sample-accurate)
        endif()
        
        # Inlining threshold (removed - not supported in Clang 20)
        
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        message(STATUS "Applying GCC optimizations")
        
        check_cxx_compiler_flag("-flto=auto" GCC_SUPPORTS_AUTO_LTO)
        if(GCC_SUPPORTS_AUTO_LTO)
            target_compile_options(${target} PRIVATE -flto=auto)
        endif()
        
        # Function inlining
        check_cxx_compiler_flag("-finline-functions" GCC_SUPPORTS_INLINE_FUNCTIONS)
        if(GCC_SUPPORTS_INLINE_FUNCTIONS)
            target_compile_options(${target} PRIVATE -finline-functions)
        endif()
        
        # Loop vectorization
        check_cxx_compiler_flag("-ftree-loop-vectorize" GCC_SUPPORTS_LOOP_VECTORIZE)
        if(GCC_SUPPORTS_LOOP_VECTORIZE)
            target_compile_options(${target} PRIVATE -ftree-loop-vectorize)
        endif()
        
        check_cxx_compiler_flag("-ftree-slp-vectorize" GCC_SUPPORTS_SLP_VECTORIZE)
        if(GCC_SUPPORTS_SLP_VECTORIZE)
            target_compile_options(${target} PRIVATE -ftree-slp-vectorize)
        endif()
        
        # Loop optimizations
        check_cxx_compiler_flag("-ftree-loop-im" GCC_SUPPORTS_LOOP_IM)
        if(GCC_SUPPORTS_LOOP_IM)
            target_compile_options(${target} PRIVATE -ftree-loop-im)
        endif()
        
        # Devirtualization
        check_cxx_compiler_flag("-fdevirtualize" GCC_SUPPORTS_DEVIRTUALIZE)
        if(GCC_SUPPORTS_DEVIRTUALIZE)
            target_compile_options(${target} PRIVATE -fdevirtualize)
        endif()
        
        check_cxx_compiler_flag("-fdevirtualize-speculatively" GCC_SUPPORTS_DEVIRT_SPEC)
        if(GCC_SUPPORTS_DEVIRT_SPEC)
            target_compile_options(${target} PRIVATE -fdevirtualize-speculatively)
        endif()
        
    endif()
endfunction()

if(BUILD_CPP)
    apply_optimizations(_event_signing_impl)
    install(TARGETS _event_signing_impl LIBRARY DESTINATION matrices_evolved RENAME cpp${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

# Rust PyO3 build
if(BUILD_RUST)
    find_program(CARGO_EXECUTABLE cargo REQUIRED)
    
    set(RUST_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/rust_target)
    set(RUST_OUTPUT_FILE ${RUST_TARGET_DIR}/release/libmatrices_evolved_rust.so)
    set(RUST_DEST_FILE ${CMAKE_CURRENT_BINARY_DIR}/matrices_evolved_rust.so)
    
    # Collect all Rust source files for dependency tracking
    file(GLOB_RECURSE RUST_SOURCES 
        ${CMAKE_CURRENT_SOURCE_DIR}/rust/src/*.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/rust/crypto/*.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.toml
        ${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.lock
    )
    
    add_custom_command(
        OUTPUT ${RUST_OUTPUT_FILE}
        COMMAND ${CMAKE_COMMAND} -E env 
            CARGO_TARGET_DIR=${RUST_TARGET_DIR}
            ${CARGO_EXECUTABLE} build --release --manifest-path=${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.toml
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rust
        DEPENDS ${RUST_SOURCES}
        COMMENT "Building Rust PyO3 module"
    )
    
    add_custom_command(
        OUTPUT ${RUST_DEST_FILE}
        COMMAND ${CMAKE_COMMAND} -E copy ${RUST_OUTPUT_FILE} ${RUST_DEST_FILE}
        DEPENDS ${RUST_OUTPUT_FILE}
        COMMENT "Copying Rust module for packaging"
    )
    
    add_custom_target(rust_build ALL DEPENDS ${RUST_DEST_FILE})
    
    # Install Rust module
    install(FILES ${RUST_DEST_FILE}
            DESTINATION matrices_evolved
            RENAME matrices_evolved_rust.so)
endif()