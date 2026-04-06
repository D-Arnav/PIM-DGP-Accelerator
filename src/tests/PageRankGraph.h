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

class PageRankGraph
{
  public:
    explicit PageRankGraph(int N)
        : N_(N), out_adj_(N), in_adj_(N), out_deg_(N, 0) {}

    int numVertices() const { return N_; }
    int outDegree(int u) const { return out_deg_[u]; }
    const vector<int>& outNeighbors(int u) const { return out_adj_[u]; }

    void addEdge(int u, int v)
    {
        for (int w : out_adj_[u])
            if (w == v) return;
        out_adj_[u].push_back(v);
        in_adj_[v].push_back(u);
        out_deg_[u]++;
    }

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

    // ================= CPU PageRank =================
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

            float diff = 0.0f;
            for (int v = 0; v < N_; v++)
                diff += fabs(new_rank[v] - rank[v]);

            swap(rank, new_rank);
            if (diff < tol) break;
        }
        return rank;
    }

    // ================= FIXED PIM MATRIX =================
    void buildTransitionMatrix(NumpyBurstType& weight_npbst,
                               float damping = 0.85f) const
    {
        int padded_N = ((N_ + 15) / 16) * 16;

        vector<vector<float>> M(N_, vector<float>(N_, 0.0f));
        float base = (1.0f - damping) / N_;

        for (int u = 0; u < N_; u++)
        {
            if (out_deg_[u] == 0)
            {
                float val = damping * (1.0f / N_);
                for (int v = 0; v < N_; v++)
                    M[v][u] = base + val;
            }
            else
            {
                for (int v = 0; v < N_; v++)
                    M[v][u] = base;

                float share = damping / out_deg_[u];
                for (int v : out_adj_[u])
                    M[v][u] += share;
            }
        }

        // ---------------- PACK ----------------
        weight_npbst.shape = {(unsigned long)N_, (unsigned long)N_};
        weight_npbst.loadTobShape(16.0);

        int bursts_per_row = padded_N / 16;
        weight_npbst.bData.resize(N_ * bursts_per_row);

        for (int row = 0; row < N_; row++)
        {
            for (int col = 0; col < padded_N; col++)
            {
                int burst_idx = row * bursts_per_row + col / 16;
                int lane = col % 16;

                float val = (col < N_) ? M[row][col] : 0.0f;
                weight_npbst.bData[burst_idx].fp16Data_[lane] =
                    convertF2H(val);
            }
        }
    }

    void buildRankVector(const vector<float>& rank,
                         NumpyBurstType& input_npbst) const
    {
        int padded_N = ((N_ + 15) / 16) * 16;

        input_npbst.shape = {1, (unsigned long)N_};
        input_npbst.loadTobShape(16.0);

        input_npbst.bData.resize(padded_N / 16);

        for (int col = 0; col < padded_N; col++)
        {
            int burst_idx = col / 16;
            int lane = col % 16;

            float val = (col < N_) ? rank[col] : 0.0f;
            input_npbst.bData[burst_idx].fp16Data_[lane] =
                convertF2H(val);
        }
    }

    void buildEdgeList(vector<pair<int,int>>& edges) const
{
    edges.clear();
    for (int u = 0; u < N_; u++)
    {
        for (int v : out_adj_[u])
        {
            edges.emplace_back(u, v);
        }
    }
}
    vector<float> getIncrementalDelta(const vector<float>& prev_rank, float damping = 0.85f) const
    {
        vector<float> next_rank(N_, 0.0f);
        const float base = (1.0f - damping) / N_;

        // 1. Compute dangling contributions
        float dangling_sum = 0.0f;
        for (int u = 0; u < N_; u++) {
            if (out_deg_[u] == 0) dangling_sum += prev_rank[u];
        }

        float dangling_contrib = damping * dangling_sum / N_;
        fill(next_rank.begin(), next_rank.end(), base + dangling_contrib);

        // 2. Propagate rank along edges
        for (int u = 0; u < N_; u++) {
            if (out_deg_[u] == 0) continue;
            float share = damping * prev_rank[u] / out_deg_[u];
            for (int v : out_adj_[u]) next_rank[v] += share;
        }

        // 3. Calculate and return the difference (Delta R)
        vector<float> delta(N_, 0.0f);
        for (int i = 0; i < N_; i++) {
            delta[i] = next_rank[i] - prev_rank[i];
        }

        return delta;
    }

  private:
    int N_;
    vector<vector<int>> out_adj_;
    vector<vector<int>> in_adj_;
    vector<int> out_deg_;
};

#endif