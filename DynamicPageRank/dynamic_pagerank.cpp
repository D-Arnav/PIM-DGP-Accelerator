#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>

// ─────────────────────────────────────────────
//  Dynamic PageRank
//  Supports: add/remove edges, add/remove nodes
//  Uses incremental residual-based propagation
// ─────────────────────────────────────────────

class DynamicPageRank {
public:
    // Damping factor (standard = 0.85)
    double d;
    // Convergence threshold
    double epsilon;

    // Adjacency: out_edges[u] = {v, ...}  (u → v)
    std::unordered_map<int, std::unordered_set<int>> out_edges;
    // Reverse adjacency: in_edges[v] = {u, ...}  (u → v)
    std::unordered_map<int, std::unordered_set<int>> in_edges;

    // Current PageRank scores
    std::unordered_map<int, double> rank;
    // Residuals: pending score to be propagated
    std::unordered_map<int, double> residual;

    int num_nodes = 0;

    DynamicPageRank(double damping = 0.85, double eps = 1e-6)
        : d(damping), epsilon(eps) {}

    // ── Node management ──────────────────────

    void add_node(int u) {
        if (out_edges.count(u)) return;
        out_edges[u];
        in_edges[u];
        num_nodes++;
        rank[u] = 1.0 / num_nodes;
        residual[u] = 0.0;
        // Re-normalise existing ranks
        _renormalise();
    }

    void remove_node(int u) {
        if (!out_edges.count(u)) return;
        // Remove all edges involving u
        for (int v : out_edges[u]) in_edges[v].erase(u);
        for (int v : in_edges[u])  out_edges[v].erase(u);
        out_edges.erase(u);
        in_edges.erase(u);
        rank.erase(u);
        residual.erase(u);
        num_nodes--;
        if (num_nodes > 0) _renormalise();
    }

    // ── Edge management ──────────────────────

    void add_edge(int u, int v) {
        add_node(u);
        add_node(v);
        if (out_edges[u].count(v)) return;   // already exists

        out_edges[u].insert(v);
        in_edges[v].insert(u);

        // Adding u→v pushes extra rank from u to v
        // Inject a residual at v proportional to u's current rank
        double delta = d * rank[u] / out_edges[u].size();
        residual[v] += delta;

        _propagate();
    }

    void remove_edge(int u, int v) {
        if (!out_edges.count(u) || !out_edges[u].count(v)) return;

        // Before removing, compute the score that was flowing u→v
        double flow = (out_edges[u].size() > 0)
                      ? d * rank[u] / out_edges[u].size()
                      : 0.0;

        out_edges[u].erase(v);
        in_edges[v].erase(u);

        // v loses that flow — inject negative residual
        residual[v] -= flow;
        _propagate();
    }

    // ── Full recompute (power iteration) ─────
    //  Use this to initialise from a batch of edges

    void full_recompute(int max_iter = 100) {
        if (num_nodes == 0) return;
        double base = 1.0 - d;
        for (auto& [u, r] : rank) r = 1.0 / num_nodes;

        for (int iter = 0; iter < max_iter; ++iter) {
            std::unordered_map<int, double> new_rank;
            double dangling = 0.0;

            for (auto& [u, _] : rank) {
                if (out_edges[u].empty())
                    dangling += rank[u];
            }

            for (auto& [u, _] : rank) {
                new_rank[u] = base + d * dangling / num_nodes;
                for (int src : in_edges[u]) {
                    new_rank[u] += d * rank[src] / out_edges[src].size();
                }
            }

            // Check convergence
            double diff = 0.0;
            for (auto& [u, r] : new_rank)
                diff += std::abs(r - rank[u]);
            rank = new_rank;
            if (diff < epsilon) break;
        }
        for (auto& [u, r] : residual) r = 0.0;
    }

    // ── Query ────────────────────────────────

    double get_rank(int u) const {
        auto it = rank.find(u);
        return (it != rank.end()) ? it->second : 0.0;
    }

    std::vector<std::pair<int,double>> top_k(int k) const {
        std::vector<std::pair<int,double>> all(rank.begin(), rank.end());
        std::partial_sort(all.begin(),
                          all.begin() + std::min(k, (int)all.size()),
                          all.end(),
                          [](auto& a, auto& b){ return a.second > b.second; });
        all.resize(std::min(k, (int)all.size()));
        return all;
    }

    void print_ranks() const {
        std::vector<std::pair<int,double>> all(rank.begin(), rank.end());
        std::sort(all.begin(), all.end(),
                  [](auto& a, auto& b){ return a.second > b.second; });
        std::cout << "\n  Node  |   PageRank\n";
        std::cout << "--------+-------------\n";
        for (auto& [u, r] : all)
            std::cout << std::setw(6) << u << "  |  "
                      << std::fixed << std::setprecision(6) << r << "\n";
    }

private:
    // Residual-based incremental propagation (like Monte-Carlo push)
    void _propagate() {
        // Priority queue: process nodes with largest |residual| first
        // to converge faster
        using PQItem = std::pair<double, int>;
        std::priority_queue<PQItem> pq;

        for (auto& [u, res] : residual)
            if (std::abs(res) > epsilon)
                pq.push({std::abs(res), u});

        while (!pq.empty()) {
            auto [abs_res, u] = pq.top(); pq.pop();

            if (std::abs(residual[u]) < epsilon) continue;

            double push = residual[u];
            rank[u]     += push;
            residual[u]  = 0.0;

            // Distribute to out-neighbours
            if (!out_edges[u].empty()) {
                double share = d * push / out_edges[u].size();
                for (int v : out_edges[u]) {
                    residual[v] += share;
                    if (std::abs(residual[v]) > epsilon)
                        pq.push({std::abs(residual[v]), v});
                }
            }
        }
    }

    void _renormalise() {
        if (num_nodes == 0) return;
        double target = 1.0 / num_nodes;
        for (auto& [u, r] : rank) r = target;
    }
};

// ─────────────────────────────────────────────
//  Demo
// ─────────────────────────────────────────────

int main() {
    std::cout << "=== Dynamic PageRank Demo ===\n\n";

    DynamicPageRank dpr(0.85, 1e-6);

    // ── Step 1: build initial graph ──
    std::cout << "Step 1: Build initial graph (nodes 0-4)\n";
    std::vector<std::pair<int,int>> initial_edges = {
        {0,1},{0,2},{1,3},{2,3},{3,4},{4,0},{2,4}
    };
    for (auto [u,v] : initial_edges) dpr.add_node(u), dpr.add_node(v);
    dpr.full_recompute();
    for (auto [u,v] : initial_edges) dpr.add_edge(u, v);
    dpr.full_recompute();   // clean baseline
    std::cout << "Ranks after initial graph:";
    dpr.print_ranks();

    // ── Step 2: add an edge dynamically ──
    std::cout << "\nStep 2: Add edge 1→4 dynamically\n";
    dpr.add_edge(1, 4);
    std::cout << "Ranks after adding 1→4:";
    dpr.print_ranks();

    // ── Step 3: remove an edge dynamically ──
    std::cout << "\nStep 3: Remove edge 4→0 dynamically\n";
    dpr.remove_edge(4, 0);
    std::cout << "Ranks after removing 4→0:";
    dpr.print_ranks();

    // ── Step 4: add a new node with edges ──
    std::cout << "\nStep 4: Add new node 5, edges 5→3 and 2→5\n";
    dpr.add_edge(5, 3);
    dpr.add_edge(2, 5);
    std::cout << "Ranks after adding node 5:";
    dpr.print_ranks();

    // ── Step 5: remove a node ──
    std::cout << "\nStep 5: Remove node 0\n";
    dpr.remove_node(0);
    std::cout << "Ranks after removing node 0:";
    dpr.print_ranks();

    // ── Top-K ──
    std::cout << "\nTop-2 nodes:\n";
    for (auto [u, r] : dpr.top_k(2))
        std::cout << "  Node " << u << " -> " << std::fixed
                  << std::setprecision(6) << r << "\n";

    return 0;
}