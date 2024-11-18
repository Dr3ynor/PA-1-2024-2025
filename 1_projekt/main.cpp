#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <mutex>

using namespace std;

int N;
double global_best_cost;
vector<int> global_best_permutation;
mutex mtx;

static bool try_load_SRLFP_file(string path_to_file, vector<vector<int>>& cost_matrix, vector<int>& out_widths)
{
    ifstream file(path_to_file);
    if (!file.is_open())
    {
        cout << "File not found" << endl;
        return false;
    }
    
    file >> N;

    out_widths = vector<int>(N);
    for (int i = 0; i < N; i++)
    {
        file >> out_widths[i];
    }
        
    cost_matrix = vector<vector<int>>(N, vector<int>(N, 0));
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < N; ++j) {
            file >> cost_matrix[i][j];
        }
    }

    file.close();

    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < i; ++j) 
        {
            cost_matrix[i][j] = cost_matrix[j][i];
        }
    }

    return true;
}

static double calculate_distance(int idx_1, int idx_2, const vector<int>& permutation, const vector<int>& widths)
{
    double distance = (widths[permutation[idx_1]] + widths[permutation[idx_2]]) / 2;

    for (int k = idx_1; k <= idx_2; ++k)
    {
        distance += widths[permutation[k]];
    }

    return distance;
}



static void create_permutation(vector<int>& permutation) 
{
    permutation.clear();
    permutation.resize(N);
    for (int i = 0; i < N; i++) 
    {
        permutation[i] = i;
    }
}

static long calculate_total_permutations() 
{
    long total_permutations = 1;
    for (int i = 2; i < N; i++) 
    {
        total_permutations *= i;
    }

    return total_permutations;
}


static double calculate_permutation_cost_BnB(const vector<vector<int>>& cost_matrix, const vector<int>& permutation, const vector<int>& widths) 
{
    double cost = 0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            cost += cost_matrix[permutation[i]][permutation[j]] * calculate_distance(i, j, permutation, widths);
            if (cost > global_best_cost) 
            {
                return -1.0 * ((i + 1) % N);
            }
        }
    }

    return cost;
}

static void process_parallel(int thread_id, int chunk_size, vector<int> permutation, const vector<vector<int>>& cost_matrix, const vector<int>& widths)
{
    for (int i = 0; i < chunk_size; i++) 
    {
        double cost = calculate_permutation_cost_BnB(cost_matrix, permutation, widths);
        if (cost < 0)
        {
            int idx = cost * -1.0;
            int bound = permutation[idx];
            int bound2 = -1;
            if (idx > 0) {
                bound2 = permutation[idx - 1];
            }

            while (next_permutation(permutation.begin() + 1,
                permutation.end())) {
                if ((bound2 == -1 && bound != permutation[idx]) ||
                    (bound != permutation[idx] || bound2 != permutation[idx - 1])) {
                    break;
                }
                // BP
                i++;
            }
        }
        else if (cost <= global_best_cost)
        {
            mtx.lock();
            global_best_cost = cost;
            global_best_permutation = permutation;
            mtx.unlock();
            next_permutation(permutation.begin() + 1, permutation.end());
        }
    }
}

static void find_minimal_permutation_serial_BnB(const vector<vector<int>>& cost_matrix, const vector<int>& widths)
{
    long total_permutations = calculate_total_permutations();
    vector<int> initial_permutation;
    create_permutation(initial_permutation);
    auto start = chrono::high_resolution_clock::now();
    for (long i = 0; i < total_permutations; i++)
    {
        double cost = calculate_permutation_cost_BnB(cost_matrix, initial_permutation, widths);
        if (cost < 0)
        {
            int idx = cost * -1.0;
            int bound = initial_permutation[idx];
            int bound2 = -1;
            if (idx > 0) {
                bound2 = initial_permutation[idx - 1];
            }

            while (next_permutation(initial_permutation.begin() + 1, initial_permutation.end())) 
            {
                if ((bound2 == -1 && bound != initial_permutation[idx]) || (bound != initial_permutation[idx] || bound2 != initial_permutation[idx - 1])) 
                {
                    break;
                }
                i++;
            }
        }
        else if (cost <= global_best_cost)
        {
            global_best_cost = cost;
            global_best_permutation = initial_permutation;
            next_permutation(initial_permutation.begin() + 1, initial_permutation.end());
        }
    }

    auto elapsed = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Elapsed: " << elapsed.count() << " us" << endl;
    cout << "Permutation cost: " << global_best_cost << endl;
    cout << "Permutation: ";
    for (auto num : global_best_permutation)
    {
        cout << num << " ";
    }
    cout << endl;
}

static void find_minimal_permutation_parallel_BnB(const vector<vector<int>>& cost_matrix, const vector<int>& widths)
{
    long total_permutations = calculate_total_permutations();

    vector<int> initial_permutation;
    create_permutation(initial_permutation);
    int worker_count = thread::hardware_concurrency();
    int chunk_size = total_permutations / worker_count;
    vector<thread> workers;

    auto start = chrono::high_resolution_clock::now();
    workers.push_back(thread(process_parallel, 0, chunk_size, initial_permutation, cost_matrix, widths));
    for (int i = 1; i < worker_count; i++)
    {
        for (int j = 0; j < chunk_size; j++)
        {
            next_permutation(initial_permutation.begin() + 1, initial_permutation.end());
        }
        workers.push_back(thread(process_parallel, i, chunk_size, initial_permutation, cost_matrix, widths));
    }

    for (auto& worker : workers)
    {
        worker.join();
    }

    auto elapsed = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Elapsed: " << elapsed.count() << " us" << endl;
    cout << "Permutation cost: " << global_best_cost << endl;
    cout << "Permutation: ";
    for (auto num : global_best_permutation)
    {
        cout << num << " ";
    }
    cout << endl;
}

int main()
{
	vector<int> widths;
	vector<vector<int>> cost_matrix;
    cout << endl << "Loading file..." << endl;
	bool isLoaded = try_load_SRLFP_file("Y-10_t.txt", cost_matrix, widths);
	if (!isLoaded)
    cout << "File not found" << endl;
        return 1;
    cout << "File loaded!" << endl;
    cout << endl << "Serial BnB" << endl;
    global_best_cost = numeric_limits<double>::max();
    find_minimal_permutation_serial_BnB(cost_matrix, widths);

    cout << endl << "Parallel BnB" << endl;
    global_best_cost = numeric_limits<double>::max();
    find_minimal_permutation_parallel_BnB(cost_matrix, widths);
    
    return 0;
}
