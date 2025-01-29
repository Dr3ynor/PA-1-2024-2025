#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <omp.h>

const double DAMPING_FACTOR = 0.85;
const double EPSILON = 1e-6;
const int MAX_ITERATIONS = 100;

void loadGraph(const std::string& filename, std::vector<std::vector<int>>& graph, int& numVertices) {
    std::ifstream file(filename);
    std::string line;
    int maxNodeId = 0;

    // Přeskočení prvních 4 řádků
    for (int i = 0; i < 4; ++i) {
        std::getline(file, line);
    }

    // Načtení hran a nalezení maximálního ID uzlu
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int fromNodeId, toNodeId;
        if (!(iss >> fromNodeId >> toNodeId)) { break; } // Chyba při čtení
        if (fromNodeId > maxNodeId) maxNodeId = fromNodeId;
        if (toNodeId > maxNodeId) maxNodeId = toNodeId;
    }

    numVertices = maxNodeId + 1;
    graph.resize(numVertices);

    // Resetování pozice v souboru
    file.clear();
    file.seekg(0, std::ios::beg);

    // Znovu přeskočení prvních 4 řádků
    for (int i = 0; i < 4; ++i) {
        std::getline(file, line);
    }

    // Načtení hran do grafu
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int fromNodeId, toNodeId;
        if (!(iss >> fromNodeId >> toNodeId)) { break; } // Chyba při čtení
        graph[fromNodeId].push_back(toNodeId);
    }
}

void pageRank(const std::vector<std::vector<int>>& graph, std::vector<double>& ranks) {
    int numVertices = graph.size();
    std::vector<double> newRanks(numVertices, 0.0);
    double delta = 1.0;
    int iteration = 0;

    while (delta > EPSILON && iteration < MAX_ITERATIONS) {
        delta = 0.0;
        #pragma omp parallel for reduction(+:delta)
        for (int i = 0; i < numVertices; ++i) {
            double newRank = (1 - DAMPING_FACTOR) / numVertices;
            for (int j = 0; j < numVertices; ++j) {
                for (int k : graph[j]) {
                    if (k == i) {
                        newRank += DAMPING_FACTOR * ranks[j] / graph[j].size();
                    }
                }
            }
            newRanks[i] = newRank;
            delta += std::abs(newRanks[i] - ranks[i]);
        }

        ranks.swap(newRanks);
        iteration++;
    }
}

int main() {
    std::vector<std::vector<int>> graph;
    int numVertices;

    loadGraph("web-BerkStan.txt", graph, numVertices);

    std::vector<double> ranks(numVertices, 1.0 / numVertices);

    pageRank(graph, ranks);

    std::cout << "Final PageRank scores for the first 10 vertices:\n";
    for (int i = 1; i <= 10; ++i) {
        std::cout << "Vertex " << i << ": " << ranks[i] << "\n";
    }

    return 0;
}