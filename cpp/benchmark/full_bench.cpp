/*

Matrices-Evolved - High Performance ops offloaded to Rust and C++
Copyright (c) 2025 Albert Blasczykowski (Aless Microsystems)

This program is licensed under the Aless Microsystems Source-Available License (Non-Commercial, No Military) v1.0 Available in the Root
Directory of the project as LICENSE in Text Format.
You may use, copy, modify, and distribute this program for Non-Commercial purposes only, subject to the terms of that license.
Use by or for military, intelligence, or defense entities or purposes is strictly prohibited.

If you distribute this program in object form or make it available to others over a network, you must provide the complete
corresponding source code for the provided functionality under this same license.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the License for details.

You should have received a copy of the License along with this program; if not, see the LICENSE file included with this source.

*/

#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <functional>
#include <map>
#include <algorithm>
#include <numeric>

#include "functions.cpp"
/// This is basically a test harness for benchmarking the various functions
/// this is not hardware independent test, results may vary accross hardware
class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        return duration.count() / 1e9;
    }
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// Matrix/Synapse realistic test data patterns
static const std::vector<uint8_t> SHA256_HASH = {
    0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,
    0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0
};

static const std::vector<uint8_t> ED25519_SIGNATURE(64, 0x01);

static const std::string EVENT_CONTENT = 
    "{\"auth_events\":[],\"content\":{\"body\":\"Hello world! This is a test message with some content.\"},"
    "\"depth\":42,\"origin_server_ts\":1234567890,\"prev_events\":[\"$prev_event_id\"],"
    "\"room_id\":\"!room:example.com\",\"sender\":\"@user:example.com\",\"type\":\"m.room.message\"}";

static const std::string DEVICE_KEY_DATA = 
    "{\"user_id\":\"@user:example.com\",\"device_id\":\"DEVICE123\","
    "\"algorithms\":[\"m.olm.v1.curve25519-aes-sha2\",\"m.megolm.v1.aes-sha2\"],"
    "\"keys\":{\"curve25519:DEVICE123\":\"curve25519_key_data\",\"ed25519:DEVICE123\":\"ed25519_key_data\"},"
    "\"signatures\":{\"@user:example.com\":{\"ed25519:DEVICE123\":\"signature_data\"}}}";

// Large sync response template for multi-MB data
static const std::string SYNC_RESPONSE_TEMPLATE = 
    "{\"next_batch\":\"s123456_789_0_1_2_3_4\",\"rooms\":{\"join\":{"
    "\"!room1:example.com\":{\"timeline\":{\"events\":["
    "{\"type\":\"m.room.message\",\"sender\":\"@user1:example.com\",\"content\":{\"body\":\"Hello world! This is a message with some content that represents typical chat data.\"},\"event_id\":\"$event1\",\"origin_server_ts\":1234567890},"
    "{\"type\":\"m.room.message\",\"sender\":\"@user2:example.com\",\"content\":{\"body\":\"Another message with different content and user data.\"},\"event_id\":\"$event2\",\"origin_server_ts\":1234567891}"
    "],\"limited\":false,\"prev_batch\":\"p123456\"}},"
    "\"!room2:example.com\":{\"timeline\":{\"events\":["
    "{\"type\":\"m.room.member\",\"sender\":\"@user3:example.com\",\"content\":{\"membership\":\"join\",\"displayname\":\"User Three\"},\"event_id\":\"$event3\",\"origin_server_ts\":1234567892}"
    "],\"limited\":false}}}}";

// Media metadata for large files
static const std::string MEDIA_METADATA = 
    "{\"content_uri\":\"mxc://example.com/media123456789\",\"content_type\":\"image/jpeg\","
    "\"size\":5242880,\"filename\":\"large_image_file.jpg\","
    "\"thumbnail_url\":\"mxc://example.com/thumb123\",\"thumbnail_info\":{\"w\":800,\"h\":600,\"size\":102400},"
    "\"blurhash\":\"LKO2?U%2Tw=w]~RBVZRi};RPxuwH\",\"xyz.amorgan.blurhash\":\"LKO2?U%2Tw=w]~RBVZRi};RPxuwH\"}";

// Data cache for reusing generated payloads
static std::map<std::pair<size_t, std::string>, std::vector<uint8_t>> data_cache;

std::vector<uint8_t> generate_data(size_t size, const std::string& type = "mixed") {
    auto cache_key = std::make_pair(size, type);
    auto it = data_cache.find(cache_key);
    if (it != data_cache.end()) {
        return it->second;
    }
    
    std::vector<uint8_t> data;
    data.reserve(size);
    
    if (type == "sha256_hash") {
        while (data.size() < size) {
            for (auto byte : SHA256_HASH) {
                if (data.size() >= size) break;
                data.push_back(byte);
            }
        }
    } else if (type == "ed25519_sig") {
        while (data.size() < size) {
            for (auto byte : ED25519_SIGNATURE) {
                if (data.size() >= size) break;
                data.push_back(byte);
            }
        }
    } else if (type == "event_content") {
        while (data.size() < size) {
            for (char c : EVENT_CONTENT) {
                if (data.size() >= size) break;
                data.push_back(static_cast<uint8_t>(c));
            }
        }
    } else if (type == "device_keys") {
        while (data.size() < size) {
            for (char c : DEVICE_KEY_DATA) {
                if (data.size() >= size) break;
                data.push_back(static_cast<uint8_t>(c));
            }
        }
    } else if (type == "sync_response") {
        while (data.size() < size) {
            for (char c : SYNC_RESPONSE_TEMPLATE) {
                if (data.size() >= size) break;
                data.push_back(static_cast<uint8_t>(c));
            }
        }
    } else if (type == "media_metadata") {
        while (data.size() < size) {
            for (char c : MEDIA_METADATA) {
                if (data.size() >= size) break;
                data.push_back(static_cast<uint8_t>(c));
            }
            for (size_t i = 0; i < 256 && data.size() < size; ++i) {
                data.push_back(static_cast<uint8_t>(i));
            }
        }
    } else {
        // For large payloads (>1MB), use hex patterns for better performance
        if (size > 1048576) {
            for (size_t i = 0; i < size; ++i) {
                data.push_back(static_cast<uint8_t>(i & 0xFF));
            }
        } else {
            for (size_t i = 0; i < size; ++i) {
                if (i % 8 == 0) {
                    data.push_back(32 + (i % 95));
                } else if (i % 8 == 1) {
                    data.push_back((i * 17) & 0xFF);
                } else if (i % 8 == 2) {
                    data.push_back(0x01);
                } else if (i % 8 == 3) {
                    const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    data.push_back(b64[i % 64]);
                } else {
                    data.push_back(((i >> 3) ^ (i << 2)) & 0xFF);
                }
            }
        }
    }
    
    data_cache[cache_key] = data;
    return data;
}

int main() {
    std::cout << "Base64 Encoder Benchmark\n";
    std::cout << "=============================================\n\n";
    std::cout << "These are synthetic, so as a rule of thumb\n";
    std::cout << "Running same executable many times will \n";
    std::cout << "Will show difference but in real world \n";
    std::cout << "+-5% is better metric for close calls. \n";


    auto stable_mean = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        if (v.size() > 4) {
            v.erase(v.begin());
            v.pop_back();
        }
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    
    const double cpu_freq_ghz = 3.0;
    
    std::vector<std::pair<std::string, size_t>> test_configs = {
        {"tiny_key", 16}, {"ed25519_key", 32}, {"signature", 64}, {"hash_data", 128},
        {"small_event", 256}, {"json_payload", 512}, {"medium_event", 1024}, {"large_json", 2048},
        {"large_event", 4096}, {"batch_small", 8192}, {"huge_batch", 16384}, {"sync_medium", 65536},
        {"sync_large", 262144}, {"mega_sync", 1048576}, {"sync_xl", 2097152}, {"giant_media", 5242880},
        {"massive_data", 10485760}, {"huge_federation", 16777216}, {"enormous_sync", 20971520},
        {"giant_backup", 33554432}, {"colossal_backup", 52428800}
    };
    
    std::vector<std::string> data_types = {"mixed", "event_content", "sync_response", "media_metadata"};
    
    std::map<std::string, std::map<std::string, std::vector<double>>> results_by_config;
    std::map<std::string, std::vector<double>> overall_times;
    
    std::vector<std::string> function_names = {
        "OpenSSL", "AMS SSE", "Mulla SSE", "AMS SSE Aligned", "AMS SSE Aligned2", "AVX2 Lemire", "AMS AVX2 Custom", "AMS NEON"
    };
    
    std::vector<std::function<std::string(const std::vector<uint8_t>&)>> functions = {
        openssl_base64_encode, fast_sse_base64_encode, fast_mula_base64_encode,
        fast_sse_base64_encode_aligned, fast_sse_base64_encode_aligned_alt, fast_avx2_base64_encode_lemire, fast_sse_base64_encode_avx,
        fast_neon_base64_encode
    };
    
    // Base64 decoder function names and implementations
    std::vector<std::string> decoder_names = {
        "Lemire AVX2", "AMS AVX2 Range", "AMS SSE Range", "AMS NEON Range"
    };
    
    std::vector<std::function<std::vector<uint8_t>(std::string_view)>> decoders = {
        fast_base64_decode_signature, fast_base64_decode_avx2_rangecmp ,fast_base64_decode_sse2_rangecmp, fast_base64_decode_neon_rangecmp
    };
    
    // Volatile sinks to prevent dead-code elimination
    volatile size_t encoder_sink = 0;
    volatile uint8_t encoder_byte_sink = 0;
    volatile size_t decoder_sink = 0;
    volatile uint8_t decoder_byte_sink = 0;
    
    for (const auto& config : test_configs) {
        const std::string& config_name = config.first;
        for (const std::string& func : function_names) {
            results_by_config[config_name][func] = std::vector<double>();
        }
    }
    
    for (const std::string& func : function_names) {
        overall_times[func] = std::vector<double>();
    }
    
    for (const auto& config : test_configs) {
        const std::string& config_name = config.first;
        size_t size = config.second;
        
        for (const std::string& data_type : data_types) {
            const auto& data = generate_data(size, data_type);
            int iterations = size < 1048576 ? std::max(10, static_cast<int>(100000 / size)) : 
                            size < 10485760 ? 10 : 1;
            
            for (size_t i = 0; i < functions.size(); ++i) {
                Timer timer;
                std::string result;
                
                for (int j = 0; j < 5; ++j) {
                    result = functions[i](data);
                }
                
                timer.start();
                for (int j = 0; j < iterations; ++j) {
                    result = functions[i](data);
                    encoder_sink += result.size();
                    if (!result.empty()) encoder_byte_sink ^= result[0];
                }
                double elapsed = timer.stop();
                
                double avg_time_us = (elapsed * 1e6) / iterations;
                
                results_by_config[config_name][function_names[i]].push_back(avg_time_us);
                overall_times[function_names[i]].push_back(avg_time_us);
            }
        }
    }
    
    std::cout << "\n=== RESULTS BY TEST CONFIGURATION ===\n";
    for (const auto& config : test_configs) {
        const std::string& config_name = config.first;
        size_t size = config.second;
        
        std::cout << "\n" << config_name << " (" << size << " bytes):\n";
        std::cout << std::string(40, '-') << "\n";
        
        for (const std::string& func : function_names) {
            std::vector<double> valid_times = results_by_config[config_name][func];
            
            if (!valid_times.empty()) {
                double mean_time = stable_mean(valid_times);
                double cycles_per_byte = (mean_time * 1e-6 * cpu_freq_ghz * 1e9) / size;
                std::cout << std::setw(15) << func 
                          << std::setw(12) << std::fixed << std::setprecision(2) << mean_time << " μs"
                          << " | " << std::fixed << std::setprecision(3) << cycles_per_byte << " cyc/B"
                          << std::endl;
            }
        }
    }
    
    std::cout << "\n\n=== OVERALL MEAN RESULTS ===\n";
    std::cout << std::setw(15) << "Function" << std::setw(15) << "Average Time" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    for (const std::string& func : function_names) {
        std::vector<double> valid_times = overall_times[func];
        
        if (!valid_times.empty()) {
            double mean_time = stable_mean(valid_times);
            std::cout << std::setw(15) << func 
                      << std::setw(12) << std::fixed << std::setprecision(2) << mean_time << " μs"
                      << std::endl;
        }
    }
    
    // Base64 Decoder Benchmarks
    std::cout << "\n\n=== BASE64 DECODER BENCHMARKS ===\n";
    std::cout << "=============================================\n\n";
    
    std::map<std::string, std::map<std::string, std::vector<double>>> decode_results;
    std::map<std::string, std::vector<double>> decode_overall_times;
    
    for (const auto& config : test_configs) {
        const std::string& config_name = config.first;
        for (const std::string& func : decoder_names) {
            decode_results[config_name][func] = std::vector<double>();
        }
    }
    
    for (const std::string& func : decoder_names) {
        decode_overall_times[func] = std::vector<double>();
    }
    
    for (const auto& config : test_configs) {
        const std::string& config_name = config.first;
        size_t size = config.second;
        
        for (const std::string& data_type : data_types) {
            const auto& data = generate_data(size, data_type);
            
            // Encode data first to get base64 string for decoding
            std::string encoded = openssl_base64_encode(data);
            
            int iterations = size < 1048576 ? std::max(10, static_cast<int>(100000 / size)) : 
                            size < 10485760 ? 10 : 1;
            
            for (size_t i = 0; i < decoders.size(); ++i) {
                Timer timer;
                std::vector<uint8_t> result;
                
                // Warmup
                for (int j = 0; j < 5; ++j) {
                    result = decoders[i](encoded);
                }
                
                timer.start();
                for (int j = 0; j < iterations; ++j) {
                    result = decoders[i](encoded);
                    decoder_sink += result.size();
                    if (!result.empty()) decoder_byte_sink ^= result[0];
                }
                double elapsed = timer.stop();
                
                double avg_time_us = (elapsed * 1e6) / iterations;
                
                decode_results[config_name][decoder_names[i]].push_back(avg_time_us);
                decode_overall_times[decoder_names[i]].push_back(avg_time_us);
            }
        }
    }
    
    std::cout << "\n=== DECODER RESULTS BY TEST CONFIGURATION ===\n";
    for (const auto& config : test_configs) {
        const std::string& config_name = config.first;
        size_t size = config.second;
        
        std::cout << "\n" << config_name << " (" << size << " bytes):\n";
        std::cout << std::string(40, '-') << "\n";
        
        for (const std::string& func : decoder_names) {
            std::vector<double> valid_times = decode_results[config_name][func];
            
            if (!valid_times.empty()) {
                double mean_time = stable_mean(valid_times);
                double cycles_per_byte = (mean_time * 1e-6 * cpu_freq_ghz * 1e9) / size;
                std::cout << std::setw(15) << func 
                          << std::setw(12) << std::fixed << std::setprecision(2) << mean_time << " μs"
                          << " | " << std::fixed << std::setprecision(3) << cycles_per_byte << " cyc/B"
                          << std::endl;
            }
        }
    }
    
    std::cout << "\n\n=== DECODER OVERALL MEAN RESULTS ===\n";
    std::cout << std::setw(15) << "Function" << std::setw(15) << "Average Time" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    for (const std::string& func : decoder_names) {
        std::vector<double> valid_times = decode_overall_times[func];
        
        if (!valid_times.empty()) {
            double mean_time = stable_mean(valid_times);
            std::cout << std::setw(15) << func 
                      << std::setw(12) << std::fixed << std::setprecision(2) << mean_time << " μs"
                      << std::endl;
        }
    }
    
    // Consume volatile sinks to ensure they're not optimized away
    std::cout << "\n[Benchmark completed - processed " << encoder_sink << "/" << decoder_sink << " total bytes]\n";
    
    return 0;
}