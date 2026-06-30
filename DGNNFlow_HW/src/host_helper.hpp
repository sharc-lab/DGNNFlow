#ifndef HOST_HELPER_HPP
#define HOST_HELPER_HPP

#include "host.hpp"
#include <vector>

struct GraphData {
    int graph_id;
    int num_nodes;
    std::vector<std::vector<float>> node_features;
    std::vector<float> results;
};

std::vector<GraphData> load_graph_data(const std::string& file_path, const std::vector<int>& graph_ids, int num_graphs);

#endif
