#include "json.hpp"
#include "host_helper.hpp"

#include <iostream>
#include <fstream>
#include <unordered_set>

using json = nlohmann::json;

std::vector<GraphData> load_graph_data(const std::string& file_path, const std::vector<int>& graph_ids, int num_graphs) {
    std::ifstream file(file_path);
    json j = json::parse(file);

    std::vector<GraphData> graphs;
    std::unordered_set<int> graph_id_set(graph_ids.begin(), graph_ids.end());

    int count = 0;
    for (const auto& graph : j) {
        int graph_id = graph["graph_id"];
        if (!graph_ids.empty() && graph_id_set.find(graph_id) == graph_id_set.end()) {
            continue; // Skip graphs not in graph_ids
        }

        GraphData data;
        data.graph_id = graph_id;
        data.num_nodes = graph["num_nodes"];
        data.node_features = graph["node_features"].get<std::vector<std::vector<float>>>();
        data.results = graph["results"].get<std::vector<float>>();
        graphs.push_back(data);

        if (graph_ids.empty() && num_graphs != -1 && ++count >= num_graphs) {
            break; // Stop if we have processed the required number of graphs
        }
    }

    file.close();

    return graphs;
}