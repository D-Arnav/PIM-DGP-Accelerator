#ifndef __PAGERANK_GRAPH_H__
#define __PAGERANK_GRAPH_H__

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "Burst.h"
#include "FP16.h"

using namespace std;
using namespace DRAMSim;

// ---------------------------------------------------------------------------
// Directed graph with support for incremental edge insertions.
// Vertices are numbered 0..N-1.
// ---------------------------------------------------------------------------
class PageRankGraph
{
  public:
    explicit PageRankGraph(int N) : N_(N), out_adj_(N), in_adj_(N), out_deg_(N, 0) {}

    int numVertices() const { return N_; }
    int outDegree(int u) const { return out_deg_[u]; }
    const vector<int>& outNeighbors(int u) const { return out_adj_[u]; }

    // Add a directed edge u -> v.  Duplicate edges are silently ignored.
    void addEdge(int u, int v)
    {
        for (int w : out_adj_[u])
            if (w == v) return;
        out_adj_[u].push_back(v);
        in_adj_[v].push_back(u);
        out_deg_[u]++;
    }

    // Generate a random Erdos-Renyi graph with ~avg_deg out-edges per vertex.
    static PageRankGraph randomGraph(int N, double avg_deg, unsigned seed = 42)
    {
        PageRankGraph g(N);
        mt19937 rng(seed);
        double p = avg_deg / (N - 1);
        bernoulli_distribution coin(p);
        for (int u = 0; u < N; u++)
            for (int v = 0; v < N; v++)
                if (u != v && coin(rng)) g.addEdge(u, v);
        return g;
    }

    // ---------------------------------------------------------------------------
    // CPU PageRank (FP32, power iteration).
    //
    // rank_new[v] = (1-d)/N + d * sum_{u: u->v} rank[u] / out_deg[u]
    // Dangling nodes (out_deg==0) distribute rank uniformly to all vertices.
    //
    // If init_rank is provided the iteration warm-starts from those values
    // (useful for incremental updates).
    // ---------------------------------------------------------------------------
    vector<float> runPageRankCPU(float damping = 0.85f, float tol = 1e-6f,
                                  int max_iter = 200,
                                  const vector<float>* init_rank = nullptr) const
    {
        vector<float> rank(N_, 1.0f / N_);
        if (init_rank && (int)init_rank->size() == N_) rank = *init_rank;

        vector<float> new_rank(N_);
        const float base = (1.0f - damping) / N_;

        for (int iter = 0; iter < max_iter; iter++)
        {
            // Dangling-node mass: vertices with no out-edges redistribute rank
            // uniformly so that the Markov chain remains ergodic.
            float dangling_sum = 0.0f;
            for (int u = 0; u < N_; u++)
                if (out_deg_[u] == 0) dangling_sum += rank[u];
            float dangling_contrib = damping * dangling_sum / N_;

            fill(new_rank.begin(), new_rank.end(), base + dangling_contrib);

            for (int u = 0; u < N_; u++)
            {
                if (out_deg_[u] == 0) continue;
                float share = damping * rank[u] / out_deg_[u];
                for (int v : out_adj_[u]) new_rank[v] += share;
            }

            // L1 convergence check
            float diff = 0.0f;
            for (int v = 0; v < N_; v++) diff += fabs(new_rank[v] - rank[v]);
            swap(rank, new_rank);
            if (diff < tol) break;
        }
        return rank;
    }

    // ---------------------------------------------------------------------------
    // Build a dense column-stochastic FP16 transition matrix as a NumpyBurstType
    // ready to be passed to PIMKernel::preloadGemv / executeGemv.
    //
    // Layout: weight_npbst.shape = {N, N}
    //         bShape = {N, N/16}   (every burst holds 16 FP16 values)
    //         Element M[row][col] is stored at:
    //           burst index = row*(N/16) + col/16,  lane = col%16
    // ---------------------------------------------------------------------------
    void buildTransitionMatrix(NumpyBurstType& weight_npbst) const
    {
        // Build dense transition matrix in FP32 first
        vector<vector<float>> M(N_, vector<float>(N_, 0.0f));
        float dangling_col = 1.0f / N_;  // column for dangling nodes

        for (int u = 0; u < N_; u++)
        {
            if (out_deg_[u] == 0)
            {
                // Distribute equally to all rows
                for (int v = 0; v < N_; v++) M[v][u] = dangling_col;
            }
            else
            {
                float share = 1.0f / out_deg_[u];
                for (int v : out_adj_[u]) M[v][u] = share;
            }
        }

        // Pack into NumpyBurstType (FP16 bursts of 16 elements)
        weight_npbst.shape = {(unsigned long)N_, (unsigned long)N_};
        weight_npbst.loadTobShape(16.0);
        int num_bursts = N_ * (N_ / 16);
        weight_npbst.bData.resize(num_bursts);

        for (int row = 0; row < N_; row++)
        {
            for (int col = 0; col < N_; col++)
            {
                int burst_idx = row * (N_ / 16) + col / 16;
                int lane = col % 16;
                weight_npbst.bData[burst_idx].fp16Data_[lane] = convertF2H(M[row][col]);
            }
        }
    }

    // Build a FP16 rank vector as a NumpyBurstType (shape {1, N}).
    void buildRankVector(const vector<float>& rank, NumpyBurstType& input_npbst) const
    {
        input_npbst.shape = {1, (unsigned long)N_};
        input_npbst.loadTobShape(16.0);
        input_npbst.bData.resize(N_ / 16);

        for (int col = 0; col < N_; col++)
        {
            int burst_idx = col / 16;
            int lane = col % 16;
            input_npbst.bData[burst_idx].fp16Data_[lane] = convertF2H(rank[col]);
        }
    }

  private:
    int N_;
    vector<vector<int>> out_adj_;
    vector<vector<int>> in_adj_;
    vector<int> out_deg_;
};

#endif  // __PAGERANK_GRAPH_H__
