#include "tb.hpp"
#include "../src/edge_conv_kernel.hpp"
#include "tb_helper.hpp"

#include <getopt.h>
#include <iostream>
#include <sstream>
#include <vector>

// Function to split a string by a delimiter and return a vector of integers
std::vector<int> split(const std::string &s, char delimiter) {
    std::vector<int> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(std::stoi(token));
    }
    return tokens;
}

// Main function
int main(int argc, char* argv[]) {
    std::string weights; // set as path to the directory weights_files_Delphes/
    std::string file_path = "graph_results_Delphes.json";
    int num_graphs_to_process = 3; // set as -1 as a flag if no value was provided
    std::vector<int> graph_indices = {0, 1, 2}; // set vector to store the indices of the graphs to process

    std::cout << "Weights path: " << weights << std::endl;
    std::cout << "Graphs file path: " << file_path << std::endl;

    // // Load all graphs
    std::vector<GraphData> graphs = load_graph_data(file_path, graph_indices, num_graphs_to_process);

    num_graphs_to_process = graphs.size();

    if (graph_indices.empty()) {
        std::cout << "Processing all graphs from 0 - " << num_graphs_to_process - 1 << std::endl;
    } else {
        std::cout << "Processing graphs: ";
        for (int i = 0; i < graph_indices.size(); ++i) {
            std::cout << graph_indices[i] << " ";
        }
        std::cout << std::endl;
    }

    // Load the float weights
    load_float_weights(weights);

    // Initialize the node embedding parameters
    initialize_node_embedding_parameters();

    // Initialize the edge convolution parameters
    initialize_edge_conv_parameters();

    // testbench results
    std::vector<std::vector<float>> tb_results;

    // dut results
    std::vector<std::vector<float>> dut_results;

    int reload_weights = 1;
    float radius= 0.4;

    // Process each graph
    for (int graph_index = 0; graph_index < num_graphs_to_process && graph_index < graphs.size(); ++graph_index) {
        const auto& graph = graphs[graph_index];

        // Inputs to Kernel
        int graph_id = graph.graph_id;
        int num_nodes = 0;
        int num_edges = 0;
        // std::vector<edge_t> edge_list;
        // std::vector<node_feature_t> node_feature;

        node_feature_t* node_feature = nullptr;
        edge_t* edge_list =  nullptr;

        // Fetch the number of nodes in the graph
        num_nodes = graph.num_nodes;

        // Create an 2d output buffer for the kernel sized by the (number of nodes, NUM_TASK), allocated memory and /initialized to zero and of type FM_TYPE
        FM_TYPE out[MAX_NODE];
        for(int i = 0; i < MAX_NODE; i++) {
            out[i] = FM_TYPE(0);
        }
    

        // Allocate memory for node features and edge list
        node_feature = new node_feature_t[num_nodes];
        edge_list = new edge_t[MAX_EDGE];

        // Try loading the node features of the current graph
        try {
            for (int i = 0; i < num_nodes; ++i) {
                const auto& features = graph.node_features[i];
                if (features.size() != ND_FEATURE) {
                    throw std::runtime_error("Mismatch in number of node features!");
                }

                // Convert each node's features to fixed-point format
                for (int j = 0; j < ND_FEATURE; ++j) {
                    node_feature[i][j] = static_cast<FM_TYPE>(features[j]);
                }
            }
        }  
        catch (const std::exception& e) {
            std::cerr << "Error processing graph: " << e.what() << std::endl;
            delete[] node_feature;
            delete[] edge_list;
            continue;
        }

        // Get the Edge List for the graph
        fetch_one_graph(node_feature, edge_list, radius, num_nodes, num_edges, MAX_EDGE);

        // Resize edge_list to the correct size
        edge_t* resized_edge_list = new edge_t[num_edges];
        std::copy(edge_list, edge_list + num_edges, resized_edge_list);
        delete[] edge_list;
        edge_list = resized_edge_list;

        // Call the Edge Convolution Kernel
        std::cout << "Processing Graph [" << graph_id << "] " << "Number of Nodes: " << graph.num_nodes << std::endl;
        edge_conv_compute_kernel(
            num_nodes,
            num_edges,
            reload_weights,
            out,
            node_feature,
            edge_list,
            graphmet_embed_charge_weight_fixed,
            graphmet_embed_pdgid_weight_fixed,
            graphmet_embed_continuous_0_weight_fixed,
            graphmet_embed_continuous_0_bias_fixed,
            graphmet_embed_categorical_0_weight_fixed,
            graphmet_embed_categorical_0_bias_fixed,
            graphmet_encode_all_weight_fixed,
            graphmet_encode_all_bias_fixed,
            graphmet_bn_all_weight_fixed,
            graphmet_bn_all_bias_fixed,
            graphmet_bn_all_running_mean_fixed,
            graphmet_bn_all_running_var_fixed,
            graphmet_edge_conv_linear_weight_fixed,
            graphmet_edge_conv_linear_bias_fixed,
            graphmet_edge_conv_batchnorm_weight_fixed,
            graphmet_edge_conv_batchnorm_bias_fixed,
            graphmet_edge_conv_batchnorm_mean_fixed,
            graphmet_edge_conv_batchnorm_variance_fixed,
            graphmet_output_0_weight_fixed,
            graphmet_output_0_bias_fixed,
            graphmet_output_2_weight_fixed,
            graphmet_output_2_bias_fixed
        );

        
        // Get the results from the kernel
        std::vector<float> results;
        results.resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            results[i] = float(out[i]);
        }

        // Add the kernel results as part of dut results
        dut_results.push_back(results);

        // Add the graph results as part of tb results
        // tb_results.push_back(graph.results);

        if (reload_weights) {
            reload_weights = 0;
        }

        // Deallocate memory
        delete[] node_feature;
        delete[] edge_list;
    }

    // Retrieve the results
    std::cout << "\nDUT result:";
    for (const auto& vec : dut_results) {
        int i = 0;
        for (float val : vec) {
            if (i % 6 == 0) printf("\n");
            else printf("\t");

            printf("Node %d: %f", i, val);
            i++;
        }
        printf("\n");
    }

    std::cout << std::endl << "-----------END----------" << std::endl;
    return 0;
}
