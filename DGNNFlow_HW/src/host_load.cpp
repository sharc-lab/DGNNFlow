#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include "host.hpp"

// WEIGHTS IN FLOATING POINT
// Weight arrays for loading node embeddings
float graphmet_embed_charge_weight_float[3][EMB_DIM / 4];
float graphmet_embed_pdgid_weight_float[7][EMB_DIM / 4];
float graphmet_embed_continuous_0_weight_float[EMB_DIM / 2][NUM_CONT_FEAT];
float graphmet_embed_continuous_0_bias_float[EMB_DIM / 2];
float graphmet_embed_categorical_0_weight_float[EMB_DIM / 2][EMB_DIM / 2];
float graphmet_embed_categorical_0_bias_float[EMB_DIM / 2];

float graphmet_encode_all_weight_float[EMB_DIM][EMB_DIM];
float graphmet_encode_all_bias_float[EMB_DIM];

float graphmet_bn_all_weight_float[EMB_DIM];
float graphmet_bn_all_bias_float[EMB_DIM];
float graphmet_bn_all_running_mean_float[EMB_DIM];
float graphmet_bn_all_running_var_float[EMB_DIM];

// Weight arrays for EdgeConv Linear Layer 1
float graphmet_conv_continuous_0_0_nn_0_weight_float[EMB_DIM][2 * EMB_DIM];
float graphmet_conv_continuous_0_0_nn_0_bias_float[EMB_DIM];

// Weight arrays for EdgeConv BatchNorm1D Layer 1
float graphmet_conv_continuous_0_1_weight_float[EMB_DIM];
float graphmet_conv_continuous_0_1_bias_float[EMB_DIM];
float graphmet_conv_continuous_0_1_running_mean_float[EMB_DIM];
float graphmet_conv_continuous_0_1_running_var_float[EMB_DIM];

// Weight arrays for EdgeConv Linear Layer 2
float graphmet_conv_continuous_1_0_nn_0_weight_float[EMB_DIM][2 * EMB_DIM];
float graphmet_conv_continuous_1_0_nn_0_bias_float[EMB_DIM];

// Weight arrays for EdgeConv BatchNorm1D Layer 2
float graphmet_conv_continuous_1_1_weight_float[EMB_DIM];
float graphmet_conv_continuous_1_1_bias_float[EMB_DIM];
float graphmet_conv_continuous_1_1_running_mean_float[EMB_DIM];
float graphmet_conv_continuous_1_1_running_var_float[EMB_DIM];

// Weight arrays for final forward layer
float graphmet_output_0_weight_float[EMB_DIM / 2][EMB_DIM];
float graphmet_output_0_bias_float[EMB_DIM / 2];
float graphmet_output_2_weight_float[1][EMB_DIM / 2];
float graphmet_output_2_bias_float[1];


/**
 * Function to load float weights
 * Description: Read weights from .bin files into float weight arrays
 */
void load_float_weights(std::string weights) {
    FILE* f;

    std::cout << "Loading float weights ..." << std::endl;

    // Helper lambda function to open files and check for errors
    auto safe_fopen = [](const std::string& file_path) -> FILE* {
        FILE* file = fopen(file_path.c_str(), "rb");
        if (!file) {
            throw std::runtime_error("Error: Unable to open file " + file_path);
        }
        return file;
    };

    // Helper lambda function to safely read data from files
    auto safe_fread = [](void* ptr, size_t size, size_t count, FILE* file, const std::string& file_path) {
        size_t read_count = fread(ptr, size, count, file);
        if (read_count != count) {
            throw std::runtime_error("Error: Unable to read the correct amount of data from " + file_path);
        }
    };

    try {
        // Load each weight file with error handling
        f = safe_fopen(weights + "graphnet_embed_charge_weight.bin");
        safe_fread(graphmet_embed_charge_weight_float, sizeof(float), 24, f, weights + "graphnet_embed_charge_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_embed_pdgid_weight.bin");
        safe_fread(graphmet_embed_pdgid_weight_float, sizeof(float), 56, f, weights + "graphnet_embed_pdgid_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_embed_continuous_0_weight.bin");
        safe_fread(graphmet_embed_continuous_0_weight_float, sizeof(float), 96, f, weights + "graphnet_embed_continuous_0_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_embed_continuous_0_bias.bin");
        safe_fread(graphmet_embed_continuous_0_bias_float, sizeof(float), 16, f, weights + "graphnet_embed_continuous_0_bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_embed_categorical_0_weight.bin");
        safe_fread(graphmet_embed_categorical_0_weight_float, sizeof(float), 256, f, weights + "graphnet_embed_categorical_0_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_embed_categorical_0_bias.bin");
        safe_fread(graphmet_embed_categorical_0_bias_float, sizeof(float), 16, f, weights + "graphnet_embed_categorical_0_bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_encode_all_0_weight.bin");
        safe_fread(graphmet_encode_all_weight_float, sizeof(float), 1024, f, weights + "graphnet_encode_all_0_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_encode_all_0_bias.bin");
        safe_fread(graphmet_encode_all_bias_float, sizeof(float), 32, f, weights + "graphnet_encode_all_0_bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_bn_all_weight.bin");
        safe_fread(graphmet_bn_all_weight_float, sizeof(float), 32, f, weights + "graphnet_bn_all_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_bn_all_bias.bin");
        safe_fread(graphmet_bn_all_bias_float, sizeof(float), 32, f, weights + "graphnet_bn_all_bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_bn_all_running_mean.bin");
        safe_fread(graphmet_bn_all_running_mean_float, sizeof(float), 32, f, weights + "graphnet_bn_all_running_mean.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_bn_all_running_var.bin");
        safe_fread(graphmet_bn_all_running_var_float, sizeof(float), 32, f, weights + "graphnet_bn_all_running_var.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_0_0_nn_0_weight.bin");
        safe_fread(graphmet_conv_continuous_0_0_nn_0_weight_float, sizeof(float), 2048, f, weights + "graphnet_conv_continuous_0_0_nn_0_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_0_0_nn_0_bias.bin");
        safe_fread(graphmet_conv_continuous_0_0_nn_0_bias_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_0_0_nn_0_bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_0_1_weight.bin");
        safe_fread(graphmet_conv_continuous_0_1_weight_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_0_1_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_0_1_bias.bin");
        safe_fread(graphmet_conv_continuous_0_1_bias_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_0_1_bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_0_1_running_mean.bin");
        safe_fread(graphmet_conv_continuous_0_1_running_mean_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_0_1_running_mean.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_0_1_running_var.bin");
        safe_fread(graphmet_conv_continuous_0_1_running_var_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_0_1_running_var.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_1_0_nn_0_weight.bin");
        safe_fread(graphmet_conv_continuous_1_0_nn_0_weight_float, sizeof(float), 2048, f, weights + "graphnet_conv_continuous_1_0_nn_0_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_1_0_nn_0_bias.bin");
        safe_fread(graphmet_conv_continuous_1_0_nn_0_bias_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_1_0_nn_0_bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_1_1_weight.bin");
        safe_fread(graphmet_conv_continuous_1_1_weight_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_1_1_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_1_1_bias.bin");
        safe_fread(graphmet_conv_continuous_1_1_bias_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_1_1_bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_1_1_running_mean.bin");
        safe_fread(graphmet_conv_continuous_1_1_running_mean_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_1_1_running_mean.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_conv_continuous_1_1_running_var.bin");
        safe_fread(graphmet_conv_continuous_1_1_running_var_float, sizeof(float), 32, f, weights + "graphnet_conv_continuous_1_1_running_var.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_output_0_weight.bin");
        safe_fread(graphmet_output_0_weight_float, sizeof(float), 512, f, weights + "graphnet_output_0_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_output_0_bias.bin");
        safe_fread(graphmet_output_0_bias_float, sizeof(float), 16, f, weights + "graphnet_output_0_bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_output_2_weight.bin");
        safe_fread(graphmet_output_2_weight_float, sizeof(float), 16, f, weights + "graphnet_output_2_weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet_output_2_bias.bin");
        safe_fread(graphmet_output_2_bias_float, sizeof(float), 1, f, weights + "graphnet_output_2_bias.bin");
        fclose(f);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        throw;
    }
}

// Function to initialize for NE parameters
void initialize_node_embedding_parameters() {

    std::cout << "Initialize node embedding parameters ..." << std::endl;

    // Convert graphmet_embed_charge_weight
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < EMB_DIM / 4; ++j) {
            graphmet_embed_charge_weight_fixed[i * (EMB_DIM / 4) + j] = static_cast<WT_TYPE>(graphmet_embed_charge_weight_float[i][j]);
        }
    }

    // Convert graphmet_embed_pdgid_weight
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < EMB_DIM / 4; ++j) {
            graphmet_embed_pdgid_weight_fixed[i * (EMB_DIM / 4) + j] = static_cast<WT_TYPE>(graphmet_embed_pdgid_weight_float[i][j]);
        }
    }

    // Convert graphmet_embed_continuous_0_weight
    for (int i = 0; i < EMB_DIM / 2; ++i) {
        for (int j = 0; j < NUM_CONT_FEAT; ++j) {
            graphmet_embed_continuous_0_weight_fixed[i * NUM_CONT_FEAT + j] = static_cast<WT_TYPE>(graphmet_embed_continuous_0_weight_float[i][j]);
        }
    }

    // Convert graphmet_embed_continuous_0_bias
    for (int i = 0; i < EMB_DIM / 2; ++i) {
        graphmet_embed_continuous_0_bias_fixed[i] = static_cast<WT_TYPE>(graphmet_embed_continuous_0_bias_float[i]);
    }

    // Convert graphmet_embed_categorical_0_weight
    for (int i = 0; i < EMB_DIM / 2; ++i) {
        for (int j = 0; j < EMB_DIM / 2; ++j) {
            graphmet_embed_categorical_0_weight_fixed[i * (EMB_DIM / 2) + j] = static_cast<WT_TYPE>(graphmet_embed_categorical_0_weight_float[i][j]);
        }
    }

    // Convert graphmet_embed_categorical_0_bias
    for (int i = 0; i < EMB_DIM / 2; ++i) {
        graphmet_embed_categorical_0_bias_fixed[i] = static_cast<WT_TYPE>(graphmet_embed_categorical_0_bias_float[i]);
    }

    // Convert graphmet_encode_all_weight
    for (int i = 0; i < EMB_DIM; ++i) {
        for (int j = 0; j < EMB_DIM; ++j) {
            graphmet_encode_all_weight_fixed[i * EMB_DIM + j] = static_cast<WT_TYPE>(graphmet_encode_all_weight_float[i][j]);
        }
    }

    // Convert graphmet_encode_all_bias
    for (int i = 0; i < EMB_DIM; ++i) {
        graphmet_encode_all_bias_fixed[i] = static_cast<WT_TYPE>(graphmet_encode_all_bias_float[i]);
    }

    // Convert Batch Norm Weights
    for (int i = 0; i < EMB_DIM; ++i) {
        graphmet_bn_all_weight_fixed[i] = static_cast<WT_TYPE>(graphmet_bn_all_weight_float[i]);
        graphmet_bn_all_bias_fixed[i] = static_cast<WT_TYPE>(graphmet_bn_all_bias_float[i]);
        graphmet_bn_all_running_mean_fixed[i] = static_cast<WT_TYPE>(graphmet_bn_all_running_mean_float[i]);
        graphmet_bn_all_running_var_fixed[i] = static_cast<WT_TYPE>(graphmet_bn_all_running_var_float[i]);
    }

    // Conversion for graphmet_output_0_weight_float
    for (int i = 0; i < EMB_DIM / 2; ++i) {
        for (int j = 0; j < EMB_DIM; ++j) {
            graphmet_output_0_weight_fixed[i * EMB_DIM + j] = static_cast<WT_TYPE>(graphmet_output_0_weight_float[i][j]);
        }
    }

    // Conversion for graphmet_output_0_bias_float
    for (int i = 0; i < EMB_DIM / 2; ++i) {
        graphmet_output_0_bias_fixed[i] = static_cast<WT_TYPE>(graphmet_output_0_bias_float[i]);
    }

    // Conversion for graphmet_output_2_weight_float
    for (int i = 0; i < EMB_DIM / 2; ++i) {
        graphmet_output_2_weight_fixed[i] = static_cast<WT_TYPE>(graphmet_output_2_weight_float[0][i]);
    }

    // Conversion for graphmet_output_2_bias_float
    graphmet_output_2_bias_fixed[0] = static_cast<WT_TYPE>(graphmet_output_2_bias_float[0]);
}


// Function to initialize for Edge Conv parameters
void initialize_edge_conv_parameters() {

    std::cout << "Initialize Edge Conv parameters ..." << std::endl;

    // Populate each parameter into the aligned_vectors
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {

        float (*weights)[2 * EMB_DIM];
        float *biases;
        float *bn_weights;
        float *bn_biases;
        float *bn_means;
        float *bn_vars;

        if (layer == 0) {
            weights = graphmet_conv_continuous_0_0_nn_0_weight_float;
            biases = graphmet_conv_continuous_0_0_nn_0_bias_float;
            bn_weights = graphmet_conv_continuous_0_1_weight_float;
            bn_biases = graphmet_conv_continuous_0_1_bias_float;
            bn_means = graphmet_conv_continuous_0_1_running_mean_float;
            bn_vars = graphmet_conv_continuous_0_1_running_var_float;
        } else {
            weights = graphmet_conv_continuous_1_0_nn_0_weight_float;
            biases = graphmet_conv_continuous_1_0_nn_0_bias_float;
            bn_weights = graphmet_conv_continuous_1_1_weight_float;
            bn_biases = graphmet_conv_continuous_1_1_bias_float;
            bn_means = graphmet_conv_continuous_1_1_running_mean_float;
            bn_vars = graphmet_conv_continuous_1_1_running_var_float;
        }

        // Convert and store linear weights
        for (int i = 0; i < EMB_DIM; ++i) {
            for (int j = 0; j < 2 * EMB_DIM; ++j) {
                graphmet_edge_conv_linear_weight_fixed[(layer * EMB_DIM + i) * (2 * EMB_DIM) + j] = static_cast<WT_TYPE>(weights[i][j]);
            }
        }

        // Convert and store linear biases
        for (int i = 0; i < EMB_DIM; ++i) {
            graphmet_edge_conv_linear_bias_fixed[layer * EMB_DIM + i] = static_cast<WT_TYPE>(biases[i]);
        }

        // Convert and store batch normalization parameters
        for (int i = 0; i < EMB_DIM; ++i) {
            graphmet_edge_conv_batchnorm_weight_fixed[layer * EMB_DIM + i] = static_cast<WT_TYPE>(bn_weights[i]);
            graphmet_edge_conv_batchnorm_bias_fixed[layer * EMB_DIM + i] = static_cast<WT_TYPE>(bn_biases[i]);
            graphmet_edge_conv_batchnorm_mean_fixed[layer * EMB_DIM + i] = static_cast<WT_TYPE>(bn_means[i]);
            graphmet_edge_conv_batchnorm_variance_fixed[layer * EMB_DIM + i] = static_cast<WT_TYPE>(bn_vars[i]);
        }
    }
}


// Helper Function (for fetch_one_graph) to calculate Euclidean Distance
float euclidean(node_feature_t p1, node_feature_t p2) {
    float eta1 = p1[3].to_float(); // Extract eta from node feature
    float phi1 = p1[4].to_float(); // Extract phi from node feature
    float eta2 = p2[3].to_float();
    float phi2 = p2[4].to_float();

    // Handle periodic boundary condition for phi (angles)
    float dphi = phi1 - phi2;
    float deta = eta1 - eta2;

    // Return Euclidean distance
    return std::sqrt((deta * deta) + (dphi * dphi));
}


// Function to fetch 1 graph
void fetch_one_graph(
    node_feature_t* node_feature,
    edge_t* edge_list,
    float radius,
    int num_nodes,
    int& num_edges,
    int max_edges
) {
    num_edges = 0;
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = i + 1; j < num_nodes; ++j) {
            // Calculate distance between node i and node j
            float distance = euclidean(node_feature[i], node_feature[j]);

            // Check if the distance is within the radius
            if (distance <= radius) {
                // Add edge (i -> j) if max_edges is not exceeded
                if (num_edges < max_edges) {
                    edge_list[num_edges++] = {i, j};
                    edge_list[num_edges++] = {j, i};
                } else {
                    num_edges += 2;
                }
            }
        }
    }
}
