/**
 * HOST CODE FOR FPGA BENCHMARKING (Latency vs Batch Size)
 * Methodology: Batch=1, Single Stream, End-to-End Latency.
 */

#include "host.hpp"
#include "host_helper.hpp"
#include "host_load.hpp"
#include "xcl2/xcl2.hpp"
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

// ------------------------------------------------------------------------------------
// GLOBAL WEIGHT BUFFERS
// ------------------------------------------------------------------------------------
aligned_vector<WT_TYPE> graphmet_embed_charge_weight_fixed(3 * (EMB_DIM / 4));
aligned_vector<WT_TYPE> graphmet_embed_pdgid_weight_fixed(7 * (EMB_DIM / 4));
aligned_vector<WT_TYPE> graphmet_embed_continuous_0_weight_fixed((EMB_DIM / 2) * NUM_CONT_FEAT);
aligned_vector<WT_TYPE> graphmet_embed_continuous_0_bias_fixed(EMB_DIM / 2);
aligned_vector<WT_TYPE> graphmet_embed_categorical_0_weight_fixed((EMB_DIM / 2) * (EMB_DIM / 2));
aligned_vector<WT_TYPE> graphmet_embed_categorical_0_bias_fixed(EMB_DIM / 2);

aligned_vector<WT_TYPE> graphmet_encode_all_weight_fixed(EMB_DIM * EMB_DIM);
aligned_vector<WT_TYPE> graphmet_encode_all_bias_fixed(EMB_DIM);

aligned_vector<WT_TYPE> graphmet_bn_all_weight_fixed(EMB_DIM);
aligned_vector<WT_TYPE> graphmet_bn_all_bias_fixed(EMB_DIM);
aligned_vector<WT_TYPE> graphmet_bn_all_running_mean_fixed(EMB_DIM);
aligned_vector<WT_TYPE> graphmet_bn_all_running_var_fixed(EMB_DIM);

aligned_vector<WT_TYPE> graphmet_edge_conv_linear_weight_fixed(NUM_LAYERS * EMB_DIM * (2 * EMB_DIM));
aligned_vector<WT_TYPE> graphmet_edge_conv_linear_bias_fixed(NUM_LAYERS * EMB_DIM);
aligned_vector<WT_TYPE> graphmet_edge_conv_batchnorm_weight_fixed(NUM_LAYERS * EMB_DIM);
aligned_vector<WT_TYPE> graphmet_edge_conv_batchnorm_bias_fixed(NUM_LAYERS * EMB_DIM);
aligned_vector<WT_TYPE> graphmet_edge_conv_batchnorm_mean_fixed(NUM_LAYERS * EMB_DIM);
aligned_vector<WT_TYPE> graphmet_edge_conv_batchnorm_variance_fixed(NUM_LAYERS * EMB_DIM);

aligned_vector<WT_TYPE> graphmet_output_0_weight_fixed((EMB_DIM / 2) * EMB_DIM);
aligned_vector<WT_TYPE> graphmet_output_0_bias_fixed(EMB_DIM / 2);
aligned_vector<WT_TYPE> graphmet_output_2_weight_fixed(EMB_DIM / 2);
aligned_vector<WT_TYPE> graphmet_output_2_bias_fixed(1);

// ------------------------------------------------------------------------------------
// HELPER STRUCTURES
// ------------------------------------------------------------------------------------

// Holds graph data in Host RAM to avoid Disk I/O during benchmark
struct PreloadedGraph {
    int num_nodes;
    int num_edges;
    // Use vectors to manage memory automatically
    aligned_vector<node_feature_t> node_features;
    aligned_vector<edge_t> edge_list;
    aligned_vector<FM_TYPE> output_placeholder; 
};

// Holds timing results for a single iteration
struct BenchmarkResult {
    double hw_latency_ms; // std::chrono
    double hw_write_ms;   // OpenCL Event
    double hw_exec_ms;    // OpenCL Event
    double hw_read_ms;    // OpenCL Event
};

// Helper to convert OpenCL event time to milliseconds
double get_event_duration(cl::Event &event) {
    cl_ulong start, end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    return (double)(end - start) / 1000000.0;
}

// ------------------------------------------------------------------------------------
// MAIN
// ------------------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    // --- CONFIGURATION ---
    const int DATASET_SIZE = 17000;
    const int NUM_ITERATIONS = 17000; // Number of graphs to process
    const float RADIUS = 0.4;
    std::string binaryFile = argv[1];
    std::string weights_path = "./testbench/weights_files_Delphes/";
    std::string results_path = "./testbench/graph_results_Delphes.json";

    // --- OPENCL SETUP ---
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_edge_conv_compute;
    cl::CommandQueue q;
    
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;

    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        // CRITICAL: Enable Profiling for Hardware Timestamps
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        
        std::cout << "Programming device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err == CL_SUCCESS) {
            OCL_CHECK(err, krnl_edge_conv_compute = cl::Kernel(program, "edge_conv_compute_kernel", &err));
            valid_device = true;
            break;
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program device." << std::endl;
        exit(EXIT_FAILURE);
    }

    // --- LOAD WEIGHTS & PARAMETERS ---
    std::cout << "Loading Weights and Parameters..." << std::endl;
    load_float_weights(weights_path);
    initialize_node_embedding_parameters();
    initialize_edge_conv_parameters();

    // Create Weight Buffers (Map Once, reused for all runs)
    // Note: Creating these with CL_MEM_USE_HOST_PTR avoids copying if aligned properly
    OCL_CHECK(err, cl::Buffer w_buf_01(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 3 * (EMB_DIM / 4) * sizeof(WT_TYPE), graphmet_embed_charge_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_02(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 7 * (EMB_DIM / 4) * sizeof(WT_TYPE), graphmet_embed_pdgid_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_03(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (EMB_DIM / 2) * NUM_CONT_FEAT * sizeof(WT_TYPE), graphmet_embed_continuous_0_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_04(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (EMB_DIM / 2) * sizeof(WT_TYPE), graphmet_embed_continuous_0_bias_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_05(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (EMB_DIM / 2) * (EMB_DIM / 2) * sizeof(WT_TYPE), graphmet_embed_categorical_0_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_06(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (EMB_DIM / 2) * sizeof(WT_TYPE), graphmet_embed_categorical_0_bias_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_07(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, EMB_DIM * EMB_DIM * sizeof(WT_TYPE), graphmet_encode_all_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_08(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, EMB_DIM * sizeof(WT_TYPE), graphmet_encode_all_bias_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_09(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, EMB_DIM * sizeof(WT_TYPE), graphmet_bn_all_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_10(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, EMB_DIM * sizeof(WT_TYPE), graphmet_bn_all_bias_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_11(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, EMB_DIM * sizeof(WT_TYPE), graphmet_bn_all_running_mean_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_12(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, EMB_DIM * sizeof(WT_TYPE), graphmet_bn_all_running_var_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_13(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, NUM_LAYERS * EMB_DIM * (2 * EMB_DIM) * sizeof(WT_TYPE), graphmet_edge_conv_linear_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_14(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, NUM_LAYERS * EMB_DIM * sizeof(WT_TYPE), graphmet_edge_conv_linear_bias_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_15(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, NUM_LAYERS * EMB_DIM * sizeof(WT_TYPE), graphmet_edge_conv_batchnorm_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_16(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, NUM_LAYERS * EMB_DIM * sizeof(WT_TYPE), graphmet_edge_conv_batchnorm_bias_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_17(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, NUM_LAYERS * EMB_DIM * sizeof(WT_TYPE), graphmet_edge_conv_batchnorm_mean_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_18(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, NUM_LAYERS * EMB_DIM * sizeof(WT_TYPE), graphmet_edge_conv_batchnorm_variance_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_19(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (EMB_DIM / 2) * EMB_DIM * sizeof(WT_TYPE), graphmet_output_0_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_20(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (EMB_DIM / 2) * sizeof(WT_TYPE), graphmet_output_0_bias_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_21(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (EMB_DIM / 2) * sizeof(WT_TYPE), graphmet_output_2_weight_fixed.data(), &err));
    OCL_CHECK(err, cl::Buffer w_buf_22(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 1 * sizeof(WT_TYPE), graphmet_output_2_bias_fixed.data(), &err));

    // --- PRE-LOAD GRAPH DATA ---
    std::cout << "Pre-loading real data from JSON..." << std::endl;
    
    // 1. Pass EMPTY vector to get ALL graphs
    std::vector<int> graph_indices = {}; 
    
    // 2. Request up to NUM_ITERATIONS graphs
    // (If file has fewer, it loads all of them)
    std::vector<GraphData> raw_graphs = load_graph_data(results_path, graph_indices, NUM_ITERATIONS);
    
    if (raw_graphs.empty()) { 
        std::cerr << "Error: No graphs loaded from " << results_path << std::endl; 
        return EXIT_FAILURE; 
    }
    
    std::cout << "Successfully loaded " << raw_graphs.size() << " graphs from file." << std::endl;
    
    // 3. Prepare the Stream Data
    std::vector<PreloadedGraph> stream_data(NUM_ITERATIONS);

    for(int i = 0; i < NUM_ITERATIONS; i++) {
        // CYCLIC ACCESS: If we want 1000 iterations but only loaded 100 graphs,
        // we reuse them: 0..99, then 0..99 again.
        const auto& src_graph = raw_graphs[i % raw_graphs.size()];
        
        stream_data[i].num_nodes = src_graph.num_nodes;
        stream_data[i].node_features.resize(src_graph.num_nodes);
        
        // Convert feature types
        node_feature_t* temp_feat = new node_feature_t[src_graph.num_nodes];
        edge_t* temp_edge = new edge_t[MAX_EDGE];
        int temp_num_edges = 0;

        for (int n = 0; n < src_graph.num_nodes; ++n) {
            for (int f = 0; f < ND_FEATURE; ++f) {
                temp_feat[n][f] = static_cast<FM_TYPE>(src_graph.node_features[n][f]);
                stream_data[i].node_features[n][f] = temp_feat[n][f];
            }
        }
        
        // Generate Edge List for this specific graph
        fetch_one_graph(temp_feat, temp_edge, RADIUS, src_graph.num_nodes, temp_num_edges, MAX_EDGE);
        
        stream_data[i].num_edges = temp_num_edges;
        stream_data[i].edge_list.resize(temp_num_edges);
        for(int e = 0; e < temp_num_edges; e++) {
            stream_data[i].edge_list[e] = temp_edge[e];
        }
        
        stream_data[i].output_placeholder.resize(MAX_NODE);

        delete[] temp_feat;
        delete[] temp_edge;
    }

    // --- WARMUP RUN (LOAD WEIGHTS) ---
    std::cout << "Starting Warmup (Loading Weights to FPGA)..." << std::endl;
    {
        // Use the first preloaded graph for warmup
        cl::Buffer warmup_node_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, stream_data[0].num_nodes * sizeof(node_feature_t), stream_data[0].node_features.data(), &err);
        cl::Buffer warmup_edge_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, stream_data[0].num_edges * sizeof(edge_t), stream_data[0].edge_list.data(), &err);
        cl::Buffer warmup_out_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, MAX_NODE * sizeof(FM_TYPE), stream_data[0].output_placeholder.data(), &err);

        int idx = 0;
        krnl_edge_conv_compute.setArg(idx++, stream_data[0].num_nodes);
        krnl_edge_conv_compute.setArg(idx++, stream_data[0].num_edges);
        krnl_edge_conv_compute.setArg(idx++, 1); // RELOAD_WEIGHTS = 1 (Force Load)
        krnl_edge_conv_compute.setArg(idx++, warmup_out_buf);
        krnl_edge_conv_compute.setArg(idx++, warmup_node_buf);
        krnl_edge_conv_compute.setArg(idx++, warmup_edge_buf);
        
        // Set all weights
        krnl_edge_conv_compute.setArg(idx++, w_buf_01); krnl_edge_conv_compute.setArg(idx++, w_buf_02);
        krnl_edge_conv_compute.setArg(idx++, w_buf_03); krnl_edge_conv_compute.setArg(idx++, w_buf_04);
        krnl_edge_conv_compute.setArg(idx++, w_buf_05); krnl_edge_conv_compute.setArg(idx++, w_buf_06);
        krnl_edge_conv_compute.setArg(idx++, w_buf_07); krnl_edge_conv_compute.setArg(idx++, w_buf_08);
        krnl_edge_conv_compute.setArg(idx++, w_buf_09); krnl_edge_conv_compute.setArg(idx++, w_buf_10);
        krnl_edge_conv_compute.setArg(idx++, w_buf_11); krnl_edge_conv_compute.setArg(idx++, w_buf_12);
        krnl_edge_conv_compute.setArg(idx++, w_buf_13); krnl_edge_conv_compute.setArg(idx++, w_buf_14);
        krnl_edge_conv_compute.setArg(idx++, w_buf_15); krnl_edge_conv_compute.setArg(idx++, w_buf_16);
        krnl_edge_conv_compute.setArg(idx++, w_buf_17); krnl_edge_conv_compute.setArg(idx++, w_buf_18);
        krnl_edge_conv_compute.setArg(idx++, w_buf_19); krnl_edge_conv_compute.setArg(idx++, w_buf_20);
        krnl_edge_conv_compute.setArg(idx++, w_buf_21); krnl_edge_conv_compute.setArg(idx++, w_buf_22);

        q.enqueueTask(krnl_edge_conv_compute);
        q.finish();
    }
    std::cout << "Warmup Complete. Weights Loaded." << std::endl;

    // --- BENCHMARK LOOP ---
    std::cout << "Starting Benchmark with 16K samples ..." << std::endl;
    std::vector<BenchmarkResult> results;
    results.reserve(16378);

    int startGraphIdx = 128;
    int lastGraphIdx = 16511;
    for (int i = startGraphIdx; i <= lastGraphIdx; i++) {
        
        // Events to capture Hardware Timestamps
        cl::Event ev_write, ev_kernel, ev_read;
        
        // We create buffers here to simulate "Stream Processing" where new pointers arrive.
        // Using CL_MEM_USE_HOST_PTR is fast (Zero Copy if aligned).
        OCL_CHECK(err, cl::Buffer node_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            stream_data[i].num_nodes * sizeof(node_feature_t), stream_data[i].node_features.data(), &err));
        
        if (stream_data[i].num_edges == 0 || stream_data[i].num_edges > MAX_EDGE) continue;

        OCL_CHECK(err, cl::Buffer edge_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            stream_data[i].num_edges * sizeof(edge_t), stream_data[i].edge_list.data(), &err));
            
        OCL_CHECK(err, cl::Buffer out_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            MAX_NODE * sizeof(FM_TYPE), stream_data[i].output_placeholder.data(), &err));

        // Set Arguments (Update inputs only, weights remain set)
        int idx = 0;
        krnl_edge_conv_compute.setArg(idx++, stream_data[i].num_nodes);
        krnl_edge_conv_compute.setArg(idx++, stream_data[i].num_edges);
        krnl_edge_conv_compute.setArg(idx++, 0); // RELOAD_WEIGHTS = 0 (Skip Load)
        krnl_edge_conv_compute.setArg(idx++, out_buf);
        krnl_edge_conv_compute.setArg(idx++, node_buf);
        krnl_edge_conv_compute.setArg(idx++, edge_buf);
        
        // NOTE: We do NOT set the 22 weight args again. They persist from Warmup 
        // because the kernel object retains arguments unless overwritten.
        
        // --- TIMING BLOCK START ---
        q.finish(); // Ensure idle

        // 1. Host -> Device (Input)
        // Note: No Wait List (nullptr)
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({node_buf, edge_buf}, 0, nullptr, &ev_write));
        
        // 2. Compute
        // Note: Wait for Write (ev_write)
        std::vector<cl::Event> kernel_wait = {ev_write};
        OCL_CHECK(err, err = q.enqueueTask(krnl_edge_conv_compute, &kernel_wait, &ev_kernel));
        
        // 3. Device -> Host (Output)
        // Note: Wait for Kernel (ev_kernel)
        std::vector<cl::Event> read_wait = {ev_kernel};
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST, &read_wait, &ev_read));
        
        // 4. Wait for Finish
        OCL_CHECK(err, err = q.finish());
        // --- TIMING BLOCK END ---

        BenchmarkResult res;
        res.hw_write_ms = get_event_duration(ev_write);
        res.hw_exec_ms = get_event_duration(ev_kernel);
        res.hw_read_ms = get_event_duration(ev_read);
        results.push_back(res);

        if ((i - startGraphIdx) % 100 == 0) std::cout << "Reached graph " << (i - startGraphIdx) << std::endl;
    }

    // --- REPORTING ---
    std::cout << "\n******** BENCHMARK COMPLETED ********\n" << std::endl;

    std::cout << "\n******** SAVING DATA TO CSV *********\n" << std::endl;
    // Save detailed CSV
    std::ofstream csv("LatencyDistribution.csv");
    csv << "Iteration,HW_Latency_ms,HW_Write_ms,HW_Compute_ms,HW_Read_ms\n";
    
    std::vector<double> latency_results;
    for(size_t i=0; i<results.size(); i++) {
        results[i].hw_latency_ms = results[i].hw_write_ms + results[i].hw_exec_ms + results[i].hw_read_ms;
        
        latency_results.push_back(results[i].hw_latency_ms);
        csv << i << "," 
            << results[i].hw_latency_ms << "," 
            << results[i].hw_write_ms << "," 
            << results[i].hw_exec_ms << "," 
            << results[i].hw_read_ms << "\n";
    }
    csv.close();
    std::cout << "Data saved to 'latency_distribution.csv'" << std::endl;

    // Calc Stats
    std::cout << "Number of graphs under statistics: " << latency_results.size() << std::endl;
    double sum = std::accumulate(latency_results.begin(), latency_results.end(), 0.0);
    double mean = sum / latency_results.size();
    std::sort(latency_results.begin(), latency_results.end());
    double min_val = latency_results.front();
    double max_val = latency_results.back();
    double p99 = latency_results[(int)(0.99 * latency_results.size())];
    double p50 = latency_results[(int)(0.50 * latency_results.size())];

    std::cout << "Mean Latency: " << mean << " ms" << std::endl;
    std::cout << "Min  Latency: " << min_val << " ms" << std::endl;
    std::cout << "Max  Latency: " << max_val << " ms" << std::endl;
    std::cout << "p50  Latency: " << p50 << " ms" << std::endl;
    std::cout << "p99  Latency: " << p99 << " ms" << std::endl;

    // Save Stats
    std::ofstream stats("LatencyStats.csv");
    stats << "Mean_ms,Min_ms,Max_ms,p50_ms,p99_ms\n";
    stats << mean << ","
          << min_val << ","
          << max_val << ","
          << p50 << ","
          << p99 << "\n";

    return 0;
}