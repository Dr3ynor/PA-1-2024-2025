#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <omp.h>

const double DAMPING_FACTOR = 0.85;
const double EPSILON = 1e-6;
const int MAX_ITERATIONS = 100;

void loadGraph(const std::string& filename, std::vector<std::vector<int>>& graph, std::vector<int>& outDegree, int& numVertices) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    std::string line;
    int maxNodeId = 0;
    std::vector<std::pair<int, int>> edges;
    int numEdges = 0;

    // Přeskočení prvních 4 řádků
    for (int i = 0; i < 4; ++i) {
        std::getline(file, line);
    }

    // Načítání souboru do vektoru
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    // Paralelní zpracování řádků
    #pragma omp parallel
    {
        std::vector<std::pair<int, int>> localEdges;
        int localMaxNodeId = 0;

        #pragma omp for nowait
        for (size_t i = 0; i < lines.size(); ++i) {
            std::istringstream iss(lines[i]);
            int fromNodeId, toNodeId;
            if (!(iss >> fromNodeId >> toNodeId)) continue;

            localEdges.emplace_back(fromNodeId, toNodeId);
            localMaxNodeId = std::max(localMaxNodeId, fromNodeId);
            localMaxNodeId = std::max(localMaxNodeId, toNodeId);
        }

        // Kritická sekce pro bezpečné sloučení dat
        #pragma omp critical
        {
            edges.insert(edges.end(), localEdges.begin(), localEdges.end());
            maxNodeId = std::max(maxNodeId, localMaxNodeId);
        }
    }

    numVertices = maxNodeId + 1;
    graph.resize(numVertices);
    outDegree.resize(numVertices, 0);

    // Naplnění grafu
    for (const auto& edge : edges) {
        graph[edge.first].push_back(edge.second);
        outDegree[edge.first]++;
    }

    numEdges = edges.size();
    std::cout << "Loaded " << numEdges << " edges." << std::endl;
}


void pageRank(const std::vector<std::vector<int>>& graph, const std::vector<int>& outDegree, std::vector<double>& ranks) {
    int numVertices = graph.size();
    std::vector<double> newRanks(numVertices, 0.0);
    double delta = 1.0;
    int iteration = 0;
    printf("--------------------\n");
    printf("%d\n",numVertices);
    printf("--------------------\n");
    while (delta > EPSILON && iteration < MAX_ITERATIONS) {
        delta = 0.0;

        #pragma omp parallel for reduction(+:delta)
        for (int i = 0; i < numVertices; ++i) {
            double newRank = (1 - DAMPING_FACTOR) / numVertices;
            
            for (int neighbor : graph[i]) {
                if (outDegree[neighbor] > 0) {
                    newRank += DAMPING_FACTOR * ranks[neighbor] / outDegree[neighbor];
                }
            }

            newRanks[i] = newRank;
            delta += std::fabs(newRanks[i] - ranks[i]);
        }

        ranks.swap(newRanks);
        iteration++;
        printf("Iteration %d / %d\n", iteration, MAX_ITERATIONS);
    }
}

int main() {
    std::vector<std::vector<int>> graph;
    std::vector<int> outDegree;
    int numVertices;

    loadGraph("web-BerkStan.txt", graph, outDegree, numVertices);

    std::vector<double> ranks(numVertices, 1.0 / numVertices);

    pageRank(graph, outDegree, ranks);

    std::cout << "Final PageRank scores for the first 10 vertices:\n";
    for (int i = 0; i < 10 && i < numVertices; ++i) {
        std::cout << "Vertex " << i << ": " << ranks[i] << "\n";
    }

    return 0;
}
