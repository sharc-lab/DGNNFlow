#ifndef TB_HELPER_HPP
#define TB_HELPER_HPP

#include <vector>

struct GraphData
{
    int graph_id;
    int num_nodes;
    std::vector<std::vector<float>> node_features;
    std::vector<float> results;
};

std::vector<GraphData> load_graph_data(const std::string& file_path, const std::vector<int>& graph_ids, int num_graphs_to_process);

#endif