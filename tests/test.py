#!/usr/bin/env python3
"""
Test script for matrices_evolved C++ implementation
"""

import sys
import os
import time

def test_cpp_implementation():
    """Test the C++ implementation specifically"""
    try:
        # Import the C++ wrapper module
        import matrices_evolved.cpp as cpp
        print("[PASS] C++ module imported successfully")
        
        # Test key generation
        signing_key = cpp.SigningKey.generate()
        verify_key = signing_key.get_verify_key()
        print("[PASS] Key generation works")
        
        # Test base64 encoding/decoding
        test_data = b"Hello, World!"
        encoded = cpp.encode_base64_fast(test_data)
        decoded = cpp.decode_base64_fast(encoded)
        assert decoded == test_data
        print("[PASS] Base64 encoding/decoding works")
        
        # Test JSON signing
        event = {"type": "m.room.message", "content": {"body": "Test message"}}
        signing_key_list = list(signing_key.encode())  # Convert bytes to list of ints
        signed_event = cpp.sign_json_object_fast(event, "test.com", signing_key_list, "ed25519:key1")
        print("[PASS] JSON signing works")
        
        # Test signature verification
        verify_key_list = list(verify_key.encode())  # Convert bytes to list of ints
        cpp.verify_signed_json_fast(signed_event, "test.com", verify_key_list)
        print("[PASS] Signature verification works")
        
        # Test content hash
        content_hash = cpp.compute_content_hash(event)
        print(f"[PASS] Content hash: {content_hash[0]}")
        
        print("\n[SUCCESS] All C++ tests passed!")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Failed to import C++ module: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] C++ test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rust_implementation():
    """Test the Rust implementation for comparison"""
    try:
        import matrices_evolved.rust as rust
        print("[PASS] Rust module imported successfully")
        
        # Basic test
        signing_key = rust.generate_signing_key("ed25519")
        verify_key = rust.get_verify_key(signing_key)
        print("[PASS] Rust key generation works")
        
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import Rust module: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Rust test failed: {e}")
        return False

def benchmark_implementations():
    """Simple benchmark comparison"""
    try:
        import matrices_evolved.cpp as cpp
        import matrices_evolved.rust as rust
        
        print("\nPerformance Comparison:")
        
        # Test data
        test_data = b"x" * 1000  # 1KB test data
        iterations = 1000
        
        # C++ base64 benchmark
        start = time.time()
        for _ in range(iterations):
            encoded = cpp.encode_base64_fast(test_data)
            decoded = cpp.decode_base64_fast(encoded)
        cpp_time = time.time() - start
        
        print(f"C++ base64 ({iterations} iterations): {cpp_time:.4f}s")
        
        # Rust base64 benchmark (if available)
        try:
            start = time.time()
            for _ in range(iterations):
                encoded = rust.encode_base64(test_data)
                decoded = rust.decode_base64(encoded)
            rust_time = time.time() - start
            print(f"Rust base64 ({iterations} iterations): {rust_time:.4f}s")
            
            if rust_time > 0:
                speedup = rust_time / cpp_time
                print(f"C++ is {speedup:.2f}x faster than Rust for base64")
        except:
            print("Rust base64 benchmark not available")
            
    except Exception as e:
        print(f"[FAIL] Benchmark failed: {e}")

def main():
    print("Testing matrices_evolved implementations\n")
    
    # Test C++ implementation
    print("Testing C++ implementation:")
    cpp_success = test_cpp_implementation()
    
    print("\n" + "="*50 + "\n")
    
    # Test Rust implementation
    print("Testing Rust implementation:")
    rust_success = test_rust_implementation()
    
    if cpp_success and rust_success:
        print("\n" + "="*50)
        benchmark_implementations()
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"C++ implementation: {'[PASS]' if cpp_success else '[FAIL]'}")
    print(f"Rust implementation: {'[PASS]' if rust_success else '[FAIL]'}")
    
    return 0 if cpp_success else 1

if __name__ == "__main__":
    sys.exit(main())