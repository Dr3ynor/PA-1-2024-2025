#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <limits>

#include <fstream>
#include <sstream>

using namespace std;

// Function prototypes
void affinityPropagation(const vector<vector<double>>& similarities, int maxIterations, double dampingFactor);
vector<vector<double>> loadSimilarities(const string& filename);

int main() {
    string filename = "iris.csv";
    vector<vector<double>> mnistData = loadSimilarities(filename);


    int maxIterations = 50;
    double dampingFactor = 0.9;

    time_t start = time(nullptr);
    affinityPropagation(mnistData, maxIterations, dampingFactor);
    time_t end = time(nullptr);
    printf("Elapsed time: %ld seconds\n", end - start);

    return 0;
}


vector<vector<double>> loadSimilarities(const string& filename) {
    printf("Loading similarities from file %s\n", filename.c_str());
    vector<vector<double>> similarities;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return similarities;
    }

    string line;
    int lineNumber = 0; // To track the line number for better error reporting

    while (getline(file, line)) {
        lineNumber++;
        stringstream ss(line);
        vector<double> row;
        string value;

        while (getline(ss, value, ',')) {
            try {
                // Trim whitespace around the value
                value.erase(0, value.find_first_not_of(" \t\n\r"));
                value.erase(value.find_last_not_of(" \t\n\r") + 1);

                // Convert to double if non-empty
                if (!value.empty()) {
                    row.push_back(stod(value));
                } else {
                    cerr << "Warning: Empty value at line " << lineNumber << endl;
                    row.push_back(0.0); // Default to 0 for empty values
                }
            } catch (const invalid_argument&) {
                //cerr << "Error: Non-numeric value \"" << value << "\" at line " << lineNumber << endl;
                row.push_back(0.0); // Default to 0 for invalid values
            } catch (const out_of_range&) {
                cerr << "Error: Value out of range \"" << value << "\" at line " << lineNumber << endl;
                row.push_back(0.0); // Default to 0 for out-of-range values
            }
        }

        if (!row.empty()) {
            similarities.push_back(row);
        }
    }

    file.close();
    printf("Loaded %lu rows and %lu columns\n", similarities.size(), similarities.empty() ? 0 : similarities[0].size());
    return similarities;
}

void affinityPropagation(const vector<vector<double>>& similarities, int maxIterations, double dampingFactor) {
    int n = similarities.size();
    vector<vector<double>> responsibilities(n, vector<double>(n, 0.0));
    vector<vector<double>> availabilities(n, vector<double>(n, 0.0));


    for (int iter = 0; iter < maxIterations; ++iter) {
        printf("Iteration %d out of %d\n", iter, maxIterations);
        // Update responsibilities
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                double maxVal = -numeric_limits<double>::infinity();
                for (int j = 0; j < n; ++j) {
                    if (j != k) {
                        maxVal = max(maxVal, availabilities[i][j] + similarities[i][j]);
                    }
                }
                responsibilities[i][k] = dampingFactor * responsibilities[i][k] + (1 - dampingFactor) * (similarities[i][k] - maxVal);
            }
        }

        // Update availabilities
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                if (i != k) {
                    double sum = 0.0;
                    for (int j = 0; j < n; ++j) {
                        if (j != k && j != i) {
                            sum += max(0.0, responsibilities[j][k]);
                        }
                    }
                    availabilities[i][k] = dampingFactor * availabilities[i][k] + (1 - dampingFactor) * min(0.0, responsibilities[k][k] + sum);
                } else {
                    double sum = 0.0;
                    for (int j = 0; j < n; ++j) {
                        if (j != k) {
                            sum += max(0.0, responsibilities[j][k]);
                        }
                    }
                    availabilities[i][k] = dampingFactor * availabilities[i][k] + (1 - dampingFactor) * sum;
                }
            }
        }
    }

    // Output the exemplars
    for (int i = 0; i < n; ++i) {
        double maxVal = -numeric_limits<double>::infinity();
        int exemplar = -1;
        for (int k = 0; k < n; ++k) {
            double value = responsibilities[i][k] + availabilities[i][k];
            if (value > maxVal) {
                maxVal = value;
                exemplar = k;
            }
        }
        cout << "Point " << i << " is assigned to exemplar " << exemplar << endl;
    }
}