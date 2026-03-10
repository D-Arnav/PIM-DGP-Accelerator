// ===================================================================
//  Dynamic PageRank -- Real-World Benchmark
//  Dataset : SNAP HepPh Citation Network (~34K nodes, ~420K edges)
//  Measures : Top-K | Speed vs Full Recompute | Rank Delta | Memory
// ===================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <string>

// -------------------------------------------------------------------
//  Timing
// -------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
struct Timer {
    std::chrono::time_point<Clock> t0;
    void start() { t0 = Clock::now(); }
    double ms()  { return Ms(Clock::now() - t0).count(); }
};

// -------------------------------------------------------------------
//  Memory (cross-platform)
// -------------------------------------------------------------------
#if defined(__APPLE__)
  #include <mach/mach.h>
  long get_rss_kb() {
      struct mach_task_basic_info info;
      mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
      if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                    (task_info_t)&info, &count) == KERN_SUCCESS)
          return (long)(info.resident_size / 1024);
      return -1;
  }
#elif defined(__linux__)
  long get_rss_kb() {
      std::ifstream f("/proc/self/status");
      std::string line;
      while (std::getline(f, line))
          if (line.rfind("VmRSS:", 0) == 0) {
              long kb = 0;
              std::istringstream ss(line.substr(6));
              ss >> kb; return kb;
          }
      return -1;
  }
#else
  long get_rss_kb() { return -1; }
#endif

// -------------------------------------------------------------------
//  Dynamic PageRank engine
// -------------------------------------------------------------------
class DynamicPageRank {
public:
    double d, epsilon;
    std::unordered_map<int, std::unordered_set<int>> out_edges, in_edges;
    std::unordered_map<int, double> rank, residual;
    int num_nodes = 0;

    DynamicPageRank(double damping = 0.85, double eps = 1e-6)
        : d(damping), epsilon(eps) {}

    void add_node(int u) {
        if (out_edges.count(u)) return;
        out_edges[u]; in_edges[u];
        num_nodes++;
        rank[u] = 1.0 / num_nodes;
        residual[u] = 0.0;
    }

    void remove_node(int u) {
        if (!out_edges.count(u)) return;
        for (int v : out_edges[u]) in_edges[v].erase(u);
        for (int v : in_edges[u])  out_edges[v].erase(u);
        out_edges.erase(u); in_edges.erase(u);
        rank.erase(u); residual.erase(u);
        num_nodes--;
    }

    // Fast bulk load (no propagation)
    void bulk_add_edge(int u, int v) {
        add_node(u); add_node(v);
        out_edges[u].insert(v);
        in_edges[v].insert(u);
    }

    void add_edge(int u, int v) {
        add_node(u); add_node(v);
        if (out_edges[u].count(v)) return;
        out_edges[u].insert(v);
        in_edges[v].insert(u);
        double delta = d * rank[u] / out_edges[u].size();
        residual[v] += delta;
        _propagate();
    }

    void remove_edge(int u, int v) {
        if (!out_edges.count(u) || !out_edges[u].count(v)) return;
        double flow = out_edges[u].empty() ? 0.0
                      : d * rank[u] / out_edges[u].size();
        out_edges[u].erase(v);
        in_edges[v].erase(u);
        residual[v] -= flow;
        _propagate();
    }

    void full_recompute(int max_iter = 100) {
        if (num_nodes == 0) return;
        double base = 1.0 - d;
        for (auto& [u, r] : rank) r = 1.0 / num_nodes;
        for (int iter = 0; iter < max_iter; ++iter) {
            std::unordered_map<int, double> nr;
            double dangling = 0.0;
            for (auto& [u, _] : rank)
                if (out_edges[u].empty()) dangling += rank[u];
            for (auto& [u, _] : rank) {
                nr[u] = base + d * dangling / num_nodes;
                for (int src : in_edges[u])
                    nr[u] += d * rank[src] / out_edges[src].size();
            }
            double diff = 0.0;
            for (auto& [u, r] : nr) diff += std::abs(r - rank[u]);
            rank = nr;
            if (diff < epsilon) break;
        }
        for (auto& [u, r] : residual) r = 0.0;
    }

    double get_rank(int u) const {
        auto it = rank.find(u);
        return it != rank.end() ? it->second : 0.0;
    }

    std::vector<std::pair<int,double>> top_k(int k) const {
        std::vector<std::pair<int,double>> all(rank.begin(), rank.end());
        int n = std::min(k, (int)all.size());
        std::partial_sort(all.begin(), all.begin()+n, all.end(),
                          [](auto& a, auto& b){ return a.second > b.second; });
        all.resize(n);
        return all;
    }

private:
    void _propagate() {
        using PQ = std::pair<double,int>;
        std::priority_queue<PQ> pq;
        for (auto& [u, res] : residual)
            if (std::abs(res) > epsilon) pq.push({std::abs(res), u});
        while (!pq.empty()) {
            auto [ar, u] = pq.top(); pq.pop();
            if (std::abs(residual[u]) < epsilon) continue;
            double push = residual[u];
            rank[u]    += push;
            residual[u] = 0.0;
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
};

// -------------------------------------------------------------------
//  Load SNAP edge list
// -------------------------------------------------------------------
std::vector<std::pair<int,int>> load_edges(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; exit(1); }
    std::vector<std::pair<int,int>> edges;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int u, v;
        if (ss >> u >> v) edges.push_back({u, v});
    }
    return edges;
}

// ===================================================================
//  BENCHMARK 1 -- Top-K Ranked Nodes
// ===================================================================
void bench_topk(DynamicPageRank& dpr, int k = 15) {
    std::cout << "\n+--------------------------------------+\n";
    std::cout <<   "|  BENCHMARK 1 -- Top-" << k << " Ranked Nodes  |\n";
    std::cout <<   "+--------------------------------------+\n";
    auto topk = dpr.top_k(k);
    std::cout << std::left
              << std::setw(6)  << "Rank"
              << std::setw(12) << "Node ID"
              << std::setw(14) << "PageRank" << "\n";
    std::cout << std::string(50, '-') << "\n";
    for (int i = 0; i < (int)topk.size(); ++i) {
        auto [node, score] = topk[i];
        std::cout << std::setw(6)  << (i+1)
                  << std::setw(12) << node
                  << std::fixed << std::setprecision(4) << score << "\n";
    }
}

// ===================================================================
//  BENCHMARK 2 -- Speed: Dynamic vs Full Recompute
//  Methodology: for each test edge we measure BOTH approaches on the
//  same graph state, so the comparison is apples-to-apples.
// ===================================================================
void bench_speed(DynamicPageRank& dpr,
                 const std::vector<std::pair<int,int>>& all_edges,
                 int n_updates = 20) {
    std::cout << "\n+--------------------------------------+\n";
    std::cout <<   "|  BENCHMARK 2 -- Speed Comparison     |\n";
    std::cout <<   "+--------------------------------------+\n";
    std::cout << "  (Measuring both approaches on identical graph states)\n";

    // Find n_updates edges that are NOT already in the graph,
    // so add_edge actually does work each time.
    std::vector<int> node_ids;
    for (auto& [u,_] : dpr.out_edges) node_ids.push_back(u);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, (int)node_ids.size()-1);

    std::vector<std::pair<int,int>> test_edges;
    int attempts = 0;
    while ((int)test_edges.size() < n_updates && attempts < 500000) {
        ++attempts;
        int u = node_ids[dist(rng)];
        int v = node_ids[dist(rng)];
        if (u == v || dpr.out_edges[u].count(v)) continue;
        test_edges.push_back({u, v});
    }
    int actual = (int)test_edges.size();

    // --- Measure dynamic updates ---
    Timer t;
    double dynamic_total_ms = 0.0;
    for (auto [u,v] : test_edges) {
        t.start();
        dpr.add_edge(u, v);
        dynamic_total_ms += t.ms();
        dpr.remove_edge(u, v); // restore graph state
    }

    // --- Measure full recompute for each equivalent state ---
    double full_total_ms = 0.0;
    for (auto [u,v] : test_edges) {
        dpr.add_edge(u, v);     // same state change
        t.start();
        dpr.full_recompute();   // time the recompute
        full_total_ms += t.ms();
        dpr.remove_edge(u, v); // restore graph state
    }
    // Restore clean ranks after all the add/removes
    dpr.full_recompute();

    double speedup = full_total_ms / dynamic_total_ms;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  Updates measured        : " << actual << " (genuinely new edges)\n";
    std::cout << "  Dynamic total           : " << dynamic_total_ms << " ms\n";
    std::cout << "  Full recompute total    : " << full_total_ms << " ms\n";
    std::cout << "  Speedup                 : " << speedup << "x\n";
    std::cout << "  Per dynamic update      : " << (dynamic_total_ms / actual) << " ms\n";
    std::cout << "  Per full recompute      : " << (full_total_ms    / actual) << " ms\n";
}

// ===================================================================
//  BENCHMARK 3 -- Rank Change After Genuinely New Edge Additions
// ===================================================================
void bench_rank_change(DynamicPageRank& dpr, int n_changes = 10) {
    std::cout << "\n+--------------------------------------+\n";
    std::cout <<   "|  BENCHMARK 3 -- Rank Delta After Updates |\n";
    std::cout <<   "+--------------------------------------+\n";

    // Stable baseline via full recompute
    dpr.full_recompute(100);
    auto before = dpr.top_k(5);
    std::unordered_map<int,double> before_map;
    for (auto [n,r] : before) before_map[n] = r;

    // Collect all nodes to sample from
    std::vector<int> node_ids;
    node_ids.reserve(dpr.out_edges.size());
    for (auto& [u,_] : dpr.out_edges) node_ids.push_back(u);

    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, (int)node_ids.size()-1);

    // Find edges that do NOT already exist
    std::vector<std::pair<int,int>> new_edges;
    int attempts = 0;
    while ((int)new_edges.size() < n_changes && attempts < 200000) {
        ++attempts;
        int u = node_ids[dist(rng)];
        int v = node_ids[dist(rng)];
        if (u == v) continue;
        if (dpr.out_edges[u].count(v)) continue;
        new_edges.push_back({u, v});
    }

    std::cout << "\n  Adding " << new_edges.size() << " genuinely new edges:\n";
    for (auto [u,v] : new_edges)
        std::cout << "    " << u << " -> " << v << "\n";

    for (auto [u,v] : new_edges) dpr.add_edge(u, v);
    dpr.full_recompute(100);  // stable snapshot after

    auto after = dpr.top_k(5);
    std::unordered_map<int,double> after_map;
    for (auto [n,r] : after) after_map[n] = r;

    std::unordered_set<int> nodes;
    for (auto [n,r] : before) nodes.insert(n);
    for (auto [n,r] : after)  nodes.insert(n);

    std::cout << "\n  " << std::left
              << std::setw(12) << "Node"
              << std::setw(12) << "Before"
              << std::setw(12) << "After"
              << std::setw(14) << "Delta"
              << "Movement\n";
    std::cout << "  " << std::string(58, '-') << "\n";

    for (int node : nodes) {
        double b = before_map.count(node) ? before_map[node] : dpr.get_rank(node);
        double a = after_map.count(node)  ? after_map[node]  : dpr.get_rank(node);
        double delta = a - b;
        std::string arrow = delta >  1e-7 ? "^ gained" :
                           (delta < -1e-7 ? "v lost  " : "= same  ");
        std::cout << "  " << std::setw(12) << node
                  << std::setw(12) << std::fixed << std::setprecision(4) << b
                  << std::setw(12) << a
                  << std::setw(14) << std::showpos << std::setprecision(4) << delta
                  << std::noshowpos << "  " << arrow << "\n";
    }

    // Roll back
    for (auto [u,v] : new_edges) dpr.remove_edge(u, v);
    dpr.full_recompute(100);
    std::cout << "\n  (Changes rolled back)\n";
}

// ===================================================================
//  BENCHMARK 4 -- Memory Usage
// ===================================================================
void bench_memory(DynamicPageRank& dpr) {
    std::cout << "\n+--------------------------------------+\n";
    std::cout <<   "|  BENCHMARK 4 -- Memory Usage          |\n";
    std::cout <<   "+--------------------------------------+\n";

    long rss = get_rss_kb();
    long edge_count = 0;
    for (auto& [u, outs] : dpr.out_edges) edge_count += outs.size();

    size_t rank_bytes     = dpr.rank.size()     * (sizeof(int) + sizeof(double));
    size_t residual_bytes = dpr.residual.size() * (sizeof(int) + sizeof(double));
    size_t edge_bytes     = edge_count * 2 * sizeof(int);
    size_t total_bytes    = rank_bytes + residual_bytes + edge_bytes;

    std::cout << "\n  Graph stats:\n";
    std::cout << "    Nodes           : " << dpr.num_nodes << "\n";
    std::cout << "    Edges           : " << edge_count << "\n";
    std::cout << "    Avg out-degree  : " << std::fixed << std::setprecision(2)
              << (dpr.num_nodes > 0 ? (double)edge_count / dpr.num_nodes : 0) << "\n";

    std::cout << "\n  Estimated structure sizes:\n";
    std::cout << "    rank[] array    : " << rank_bytes/1024     << " KB\n";
    std::cout << "    residual[]      : " << residual_bytes/1024 << " KB\n";
    std::cout << "    edge sets       : " << edge_bytes/1024     << " KB\n";
    std::cout << "    Total estimated : " << total_bytes/1024    << " KB  ("
              << total_bytes/1024/1024 << " MB)\n";

    if (rss > 0)
        std::cout << "    Process RSS     : " << rss << " KB  (" << rss/1024 << " MB)\n";
    else
        std::cout << "    Process RSS     : unavailable on this platform\n";

    if (dpr.num_nodes > 0)
        std::cout << "\n  Bytes per node  : " << total_bytes / dpr.num_nodes << "\n";
    if (edge_count > 0)
        std::cout << "  Bytes per edge  : " << edge_bytes / edge_count << "\n";
}

// ===================================================================
//  MAIN
// ===================================================================
int main(int argc, char* argv[]) {
    std::string path = (argc > 1) ? argv[1] : "cit-HepPh.txt";

    std::cout << "===================================================\n";
    std::cout << "  Dynamic PageRank -- HepPh Citation Network\n";
    std::cout << "===================================================\n";
    std::cout << "  Loading: " << path << "\n";

    Timer t;
    t.start();
    auto edges = load_edges(path);
    std::cout << "  Loaded " << edges.size() << " edges in "
              << std::fixed << std::setprecision(1) << t.ms() << " ms\n";

    std::cout << "\n  Building graph...\n";
    DynamicPageRank dpr(0.85, 1e-6);
    for (auto [u,v] : edges) dpr.bulk_add_edge(u, v);
    std::cout << "  Nodes: " << dpr.num_nodes << "\n";

    std::cout << "  Running initial PageRank...\n";
    t.start();
    dpr.full_recompute(100);
    std::cout << "  Done in " << t.ms() << " ms\n";

    bench_topk(dpr, 15);
    bench_speed(dpr, edges, 500);
    bench_rank_change(dpr, 10);
    bench_memory(dpr);

    std::cout << "\n===================================================\n";
    std::cout << "  All benchmarks complete.\n";
    std::cout << "===================================================\n";
    return 0;
}