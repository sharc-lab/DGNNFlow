#include "load_inputs.hpp"
#include "operators.hpp"
#include "hls_math.h"
#include "utils.hpp"

void load_weights(
    WT_TYPE graphmet_embed_charge_weight_in[3][EMB_DIM/4],
    WT_TYPE graphmet_embed_pdgid_weight_in[7][EMB_DIM/4],
    WT_TYPE graphmet_embed_continuous_0_weight_in[EMB_DIM/2][NUM_CONT_FEAT],
    WT_TYPE graphmet_embed_continuous_0_bias_in[EMB_DIM/2],
    WT_TYPE graphmet_embed_categorical_0_weight_in[EMB_DIM/2][EMB_DIM/2],
    WT_TYPE graphmet_embed_categorical_0_bias_in[EMB_DIM/2],
    WT_TYPE graphmet_encode_all_weight_in[EMB_DIM][EMB_DIM],
    WT_TYPE graphmet_encode_all_bias_in[EMB_DIM],
    WT_TYPE graphmet_bn_all_weight_in[EMB_DIM],
    WT_TYPE graphmet_bn_all_bias_in[EMB_DIM],
    WT_TYPE graphmet_bn_all_running_mean_in[EMB_DIM],
    WT_TYPE graphmet_bn_all_running_var_in[EMB_DIM],
    WT_TYPE graphmet_edge_conv_linear_weight_in[NUM_LAYERS][EMB_DIM][2*EMB_DIM],
    WT_TYPE graphmet_edge_conv_linear_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE graphmet_edge_conv_batchnorm_weight_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE graphmet_edge_conv_batchnorm_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE graphmet_edge_conv_batchnorm_mean_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE graphmet_edge_conv_batchnorm_variance_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE graphmet_output_0_weight_in[EMB_DIM/2][EMB_DIM],
    WT_TYPE graphmet_output_0_bias_in[EMB_DIM/2],
    WT_TYPE graphmet_output_2_weight_in[1][EMB_DIM/2],
    WT_TYPE graphmet_output_2_bias_in[1]
) {

    // Do not inline this function into any calling functions. It should be a separate RTL Block
    #pragma HLS INLINE off

    // Load weights for graphmet_embed_charge_weight
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < EMB_DIM / 4; ++j)
        {
            graphmet_embed_charge_weight[i][j] = graphmet_embed_charge_weight_in[i][j];
        }
    }

    // Load weights for graphmet_embed_pdgid_weight
    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < EMB_DIM / 4; ++j)
        {
            graphmet_embed_pdgid_weight[i][j] = graphmet_embed_pdgid_weight_in[i][j];
        }
    }

    // Load weights for graphmet_embed_continuous_0_weight
    for (int i = 0; i < EMB_DIM / 2; ++i)
    {
        for (int j = 0; j < NUM_CONT_FEAT; ++j)
        {
            graphmet_embed_continuous_0_weight[i][j] = graphmet_embed_continuous_0_weight_in[i][j];
        }
    }

    // Load weights for graphmet_embed_continuous_0_bias
    for (int i = 0; i < EMB_DIM / 2; ++i)
    {
        graphmet_embed_continuous_0_bias[i] = graphmet_embed_continuous_0_bias_in[i];
    }

    // Load weights for graphmet_embed_categorical_0_weight
    for (int i = 0; i < EMB_DIM / 2; ++i)
    {
        for (int j = 0; j < EMB_DIM / 2; ++j)
        {
            graphmet_embed_categorical_0_weight[i][j] = graphmet_embed_categorical_0_weight_in[i][j];
        }
    }

    // Load weights for graphmet_embed_categorical_0_bias
    for (int i = 0; i < EMB_DIM / 2; ++i)
    {
        graphmet_embed_categorical_0_bias[i] = graphmet_embed_categorical_0_bias_in[i];
    }

    // Load weights for graphmet_encode_all_weight
    for (int i = 0; i < EMB_DIM; ++i)
    {
        for (int j = 0; j < EMB_DIM; ++j)
        {
            graphmet_encode_all_weight[i][j] = graphmet_encode_all_weight_in[i][j];
        }
    }

    // Load weights for graphmet_encode_all_bias
    for (int i = 0; i < EMB_DIM; ++i)
    {
        graphmet_encode_all_bias[i] = graphmet_encode_all_bias_in[i];
    }

    // Load weights for graphmet_bn_all_weight
    for (int i = 0; i < EMB_DIM; ++i)
    {
        graphmet_bn_all_weight[i] = graphmet_bn_all_weight_in[i];
    }

    // Load weights for graphmet_bn_all_bias
    for (int i = 0; i < EMB_DIM; ++i)
    {
        graphmet_bn_all_bias[i] = graphmet_bn_all_bias_in[i];
    }

    // Load weights for graphmet_bn_all_running_mean
    for (int i = 0; i < EMB_DIM; ++i)
    {
        graphmet_bn_all_running_mean[i] = graphmet_bn_all_running_mean_in[i];
    }

    // Load weights for graphmet_bn_all_running_var
    for (int i = 0; i < EMB_DIM; ++i)
    {
        graphmet_bn_all_running_var[i] = graphmet_bn_all_running_var_in[i];
    }

    // Load weights for graphmet_output_0_weight
    for (int i = 0; i < EMB_DIM / 2; ++i)
    {
        for (int j = 0; j < EMB_DIM; ++j)
        {
            graphmet_output_0_weight[i][j] = graphmet_output_0_weight_in[i][j];
        }
    }

    // Load weights for graphmet_output_0_bias
    for (int i = 0; i < EMB_DIM / 2; ++i)
    {
        graphmet_output_0_bias[i] = graphmet_output_0_bias_in[i];
    }

    // Load weights for graphmet_output_2_weight
    for (int i = 0; i < EMB_DIM / 2; ++i)
    {
        graphmet_output_2_weight[0][i] = graphmet_output_2_weight_in[0][i];
    }

    // Load weights for graphmet_output_2_bias
    graphmet_output_2_bias[0] = graphmet_output_2_bias_in[0];

    // Load Edge Conv Linear weights
    for (int i = 0; i < NUM_LAYERS; ++i)
    {
        for (int j = 0; j < EMB_DIM; ++j)
        {
            for (int k = 0; k < 2 * EMB_DIM; k++)
            {
                // graphmet_edge_conv_linear_weight[i][j][k] = graphmet_edge_conv_linear_weight_in[i][j][k];

                // Add weights for each MP unit
                for (int mp = 0; mp < EDGE_PARALLEL; mp++)
                {
                    edge_conv_linear_weight[mp][i][j][k] = graphmet_edge_conv_linear_weight_in[i][j][k];
                }
            }
        }
    }

    // Load Edge Conv Linear bias
    for (int i = 0; i < NUM_LAYERS; ++i)
    {
        for (int j = 0; j < EMB_DIM; ++j)
        {
            // graphmet_edge_conv_linear_bias[i][j] = graphmet_edge_conv_linear_bias_in[i][j];

            // Add bias for each MP unit
            for (int mp = 0; mp < EDGE_PARALLEL; mp++)
            {
                edge_conv_linear_bias[mp][i][j] = graphmet_edge_conv_linear_bias_in[i][j];
            }
        }
    }

    // Load Edge Conv BatchNorm1D paramaters
    for (int i = 0; i < NUM_LAYERS; ++i)
    {
        for (int j = 0; j < EMB_DIM; ++j)
        {
            graphmet_edge_conv_batchnorm_weight[i][j] = graphmet_edge_conv_batchnorm_weight_in[i][j];
        }
    }

    for (int i = 0; i < NUM_LAYERS; ++i)
    {
        for (int j = 0; j < EMB_DIM; ++j)
        {
            graphmet_edge_conv_batchnorm_bias[i][j] = graphmet_edge_conv_batchnorm_bias_in[i][j];
        }
    }

    for (int i = 0; i < NUM_LAYERS; ++i)
    {
        for (int j = 0; j < EMB_DIM; ++j)
        {
            graphmet_edge_conv_batchnorm_mean[i][j] = graphmet_edge_conv_batchnorm_mean_in[i][j];
        }
    }

    for (int i = 0; i < NUM_LAYERS; ++i)
    {
        for (int j = 0; j < EMB_DIM; ++j)
        {
            graphmet_edge_conv_batchnorm_variance[i][j] = graphmet_edge_conv_batchnorm_variance_in[i][j];
        }
    }
}


/**
*
* Implementation for loading a graph
* @param   edge_list_in    // List of edges for graph connectivity
* @param   num_of_nodes    // Number of nodes in the graph
* @param   num_of_edges    // Number of edges in the graph
* @return  None
*
* Description: This function loads the graph data structures and computes the degree of each node
*/

void load_graph(
    edge_t* edge_list_in,
    int num_of_nodes,
    int num_of_edges
) {
// Do not inline this function into any calling functions. It should be a separate RTL Block
#pragma HLS INLINE off

    int neighbor_table_offset[EDGE_PARALLEL][MAX_NODE];
    
#pragma HLS ARRAY_PARTITION variable=degree_table complete dim=1
#pragma HLS ARRAY_PARTITION variable=degree_tables complete dim = 1
#pragma HLS ARRAY_PARTITION variable=neighbor_table complete dim=1
#pragma HLS ARRAY_PARTITION variable=neighbor_table_offset complete dim=1
#pragma HLS ARRAY_PARTITION variable=numEdgesPerMP complete dim=1

    // Initialize the graph data structures
    for (int node = 0; node < num_of_nodes; node++)
    {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
        degree_table[node] = 0;

        for (int mp = 0; mp < EDGE_PARALLEL; mp++)
        {
#pragma HLS UNROLL
            degree_tables[mp][node] = 0;
        }
    }

    // Count the number of edges for each node
    for (int edge = 0; edge < num_of_edges; edge++)
    {
#pragma HLS PIPELINE II=3
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_EDGE max=ANALYSIS_MAX_EDGE avg=ANALYSIS_AVG_EDGE

        // Get the edge struct from the edgeID
        // edge: u --> v
        // u : Source Node
        // v : Target Node
        edge_t edge_nd = edge_list_in[edge];

        int u = edge_nd.u;
        int v = edge_nd.v;

        degree_table[v]++;

        int mp = u % EDGE_PARALLEL;
        degree_tables[mp][v]++;

    }

    // Initialize the number of edges per table
    for ( int mp = 0; mp < EDGE_PARALLEL; mp++)
    {
#pragma HLS UNROLL
        numEdgesPerMP[mp] = 0;
    }

    // Compute the neighbor table offset
    for (int node = 0; node < num_of_nodes; node++)
    {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE

        for ( int mp = 0; mp < EDGE_PARALLEL; mp++)
        {
#pragma HLS UNROLL
            int curr_num_edges = numEdgesPerMP[mp];
            int curr_degree = degree_tables[mp][node];
            neighbor_table_offset[mp][node] = curr_num_edges;
            numEdgesPerMP[mp] = curr_num_edges + curr_degree;
        }
    }

    // Initialize the neighbor table
    for (int edge = 0; edge < num_of_edges; edge++)
    {
#pragma HLS PIPELINE II = 4
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_EDGE max=ANALYSIS_MAX_EDGE avg=ANALYSIS_AVG_EDGE

        // Get the edge struct from the edgeID
        // edge: u --> v
        // u : Source Node
        // v : Target Node
        edge_t edge_nd = edge_list_in[edge];
        int u = edge_nd.u;
        int v = edge_nd.v;
        
        int mp = u % EDGE_PARALLEL;
        int edgePtr = neighbor_table_offset[mp][v];
        neighbor_table[mp][edgePtr] = u / EDGE_PARALLEL;
        neighbor_table_offset[mp][v] = edgePtr + 1;
    }
}

/**
*
* Implementation for loading node embeddings for EdgeConv Layer
* @param   node_feature    // Vector Table of Node Features
* @param   num_of_nodes    // Number of nodes in the graph
* @return  None
*/
void load_input_node_embeddings(
    node_feature_t* node_feature,
    int num_of_nodes
) {
// Do not inline this function into any calling functions. It should be a separate RTL Block
#pragma HLS INLINE off

    FM_TYPE x_cont[MAX_NODE][NUM_CONT_FEAT];  // Continuous features (6 columns)
    FM_TYPE x_cat[MAX_NODE][NUM_CAT_FEAT];    // Categorical features (2 columns)

#pragma HLS ARRAY_PARTITION variable=x_cont complete dim=2
#pragma HLS ARRAY_PARTITION variable=x_cat complete dim=2
#pragma HLS ARRAY_PARTITION variable=norm complete dim=1

    for (int i = 0; i < num_of_nodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
// #pragma HLS PIPELINE II = 1

        // Get the vector of node features of node i
        node_feature_t node_feature_nd = node_feature[i];

        // Extract continuous features (first 6 features)
        for (int j = 0; j < NUM_CONT_FEAT; ++j)
        {
            // node_feature_nd[i][0] to node_feature_nd[i][5] are continuous features
            x_cont[i][j] = node_feature_nd[j];
        }

        // Extract categorical features (last 2 features)
        for (int j = 0; j < NUM_CAT_FEAT; ++j)
        {
            // node_feature_nd[i][6] to node_feature_nd[i][7] are categorical features
            x_cat[i][j] = node_feature_nd[NUM_CONT_FEAT + j];
        }
    }

    // x_cont *= self.datanorm
    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
#pragma HLS PIPELINE II = 2
        for (int j = 0; j < NUM_CONT_FEAT; j++)
        {
            x_cont[i][j] *= norm[j];
        }
    }

    // emb_cont = self.embed_continuous(x_cont)
    WT_TYPE emb_cont[MAX_NODE][EMB_DIM/2];
#pragma HLS ARRAY_PARTITION variable=emb_cont complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_embed_continuous_0_weight dim=2 complete

    for (int i = 0; i < num_of_nodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < EMB_DIM/2; ++j)
        {
            FM_TYPE temp = graphmet_embed_continuous_0_bias[j];
            for (int k = 0; k < NUM_CONT_FEAT; ++k)
            {
                temp += graphmet_embed_continuous_0_weight[j][k] * x_cont[i][k];
            }
            // Apply ELU activation
            // FM_TYPE temp = emb_cont[i][j];
            emb_cont[i][j] = (temp > 0) ? temp : static_cast<FM_TYPE>(hls::exp(temp) - 1);
        }
    }

    // Embed charge
    WT_TYPE emb_charge[MAX_NODE][EMB_DIM/4];
#pragma HLS ARRAY_PARTITION variable=emb_charge complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_embed_charge_weight complete dim=2
    for (int i = 0; i < num_of_nodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
#pragma HLS PIPELINE II = 1
        int charge_idx = x_cat[i][1] + 1; // Assuming charge is in the last column
        for (int j = 0; j < EMB_DIM/4; ++j)
        {
            emb_charge[i][j] = graphmet_embed_charge_weight[charge_idx][j];
        }
    }


    // Remap PDG ID and embed
    int pdg_remap[MAX_NODE];
#pragma HLS ARRAY_PARTITION variable=pdg_remap complete dim=1
    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
    // #pragma HLS PIPELINE II = 1
        pdg_remap[i] = hls::abs(x_cat[i][0]);
    }

    /**
     *  for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
    */
    for (int i = 0; i < NUM_PDGS; i++)
    {
// #pragma HLS PIPELINE II = 1
        int pdgval = pdgs[i];
        for (int row = 0; row < num_of_nodes; row++)
        {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
// #pragma HLS UNROLL factor = 8
            if (pdg_remap[row] == pdgval)
            {
                pdg_remap[row] = i;
            }
        }
    }

    /* emb_pdg = self.embed_pdgid(pdg_remap) */
    WT_TYPE emb_pdg[MAX_NODE][EMB_DIM/4];
#pragma HLS ARRAY_PARTITION variable=emb_pdg complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_embed_pdgid_weight complete dim=2
    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
#pragma HLS PIPELINE II = 1
        int idx = pdg_remap[i];

        for (int j = 0; j < EMB_DIM/4; j++)
        {
// #pragma HLS UNROLL factor=4
            emb_pdg[i][j] = graphmet_embed_pdgid_weight[idx][j];
        }
    }

    // Concatenate categorical features and pass through MLP
    FM_TYPE emb_cat_input[MAX_NODE][EMB_DIM / 2];
#pragma HLS ARRAY_PARTITION variable=emb_cat_input complete dim=2
    for (int i = 0; i < num_of_nodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < EMB_DIM / 4; ++j)
        {
            emb_cat_input[i][j] = emb_charge[i][j];
            emb_cat_input[i][j + EMB_DIM / 4] = emb_pdg[i][j];
        }
    }

    // emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg], dim=1))
    WT_TYPE emb_cat[MAX_NODE][EMB_DIM/2];
#pragma HLS ARRAY_PARTITION variable=emb_cat complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_embed_categorical_0_weight complete dim=2
    for (int i = 0; i < num_of_nodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
        for (int j = 0; j < EMB_DIM/2; ++j)
        {
            FM_TYPE acc = graphmet_embed_categorical_0_bias[j];
            for (int k = 0; k < EMB_DIM/2; ++k)
            {
                acc += graphmet_embed_categorical_0_weight[j][k] * emb_cat_input[i][k];
            }
            // emb_cat[i][j] = elu(emb_cat[i][j]);
            emb_cat[i][j] = acc > 0 ? acc : static_cast<FM_TYPE>(hls::exp(acc) - 1);
        }
    }

    // self.encode_all(torch.cat([emb_cat, emb_cont], dim=1))
    // Shape: (MAX_NODE, EMB_DIM) * (EMB_DIM, EMB_DIM) = (MAX_NODE, EMB_DIM)
// Concatenation for encode_all
    WT_TYPE encode_all_input[MAX_NODE][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=encode_all_input complete dim=2

    for (int i = 0; i < num_of_nodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
#pragma HLS PIPELINE II=1
        for (int j = 0; j < EMB_DIM/2; ++j)
        {
            encode_all_input[i][j] = emb_cat[i][j];
            encode_all_input[i][j + EMB_DIM/2] = emb_cont[i][j];
        }
    }

    // Final encoding with local buffering
    WT_TYPE encode_all[MAX_NODE][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=encode_all complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_encode_all_weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_encode_all_bias complete dim=1 

    for (int i = 0; i < num_of_nodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
        // Compute
        for (int j = 0; j < EMB_DIM; ++j)
        {
#pragma HLS PIPELINE II=1
            WT_TYPE acc = graphmet_encode_all_bias[j];
            for (int k = 0; k < EMB_DIM; ++k)
            {
                acc += graphmet_encode_all_weight[j][k] * encode_all_input[i][k];
            }
            encode_all[i][j] = acc > 0 ? acc : static_cast<WT_TYPE>(hls::exp(acc) - 1);
        }
    }

#pragma HLS ARRAY_PARTITION variable=currentNodeEmbeddings complete dim=1
#pragma HLS ARRAY_PARTITION variable=currentNodeEmbeddings complete dim = 3
#pragma HLS ARRAY_PARTITION variable=nextNodeEmbeddings complete dim = 1
#pragma HLS ARRAY_PARTITION variable=nextNodeEmbeddings complete dim = 3
#pragma HLS ARRAY_PARTITION variable=duplicateNodeEmbeddings complete dim = 1
#pragma HLS ARRAY_PARTITION variable=duplicateNodeEmbeddings complete dim = 3

#pragma HLS ARRAY_PARTITION variable=graphmet_bn_all_running_mean complete dim=1
#pragma HLS ARRAY_PARTITION variable=graphmet_bn_all_running_var complete dim=1
#pragma HLS ARRAY_PARTITION variable=graphmet_bn_all_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=graphmet_bn_all_bias complete dim=1

    for (int i = 0; i < num_of_nodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
#pragma HLS PIPELINE II=1
        for (int j = 0; j < EMB_DIM; ++j)
        {
            WT_TYPE normalized = (encode_all[i][j] - graphmet_bn_all_running_mean[j]) / hls::sqrt(graphmet_bn_all_running_var[j] + EPSILON);
            currentNodeEmbeddings[i % EDGE_PARALLEL][i / EDGE_PARALLEL][j] = (normalized * graphmet_bn_all_weight[j]) + graphmet_bn_all_bias[j];
        }
    }
}