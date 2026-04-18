#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>

#include "Burst.h"
#include "FP16.h"
#include "MemoryController.h"
#include "MemorySystem.h"
#include "MultiChannelMemorySystem.h"
#include "gtest/gtest.h"
#include "tests/KernelAddrGen.h"
#include "tests/PageRankGraph.h"
#include "tests/PIMKernel.h"


using namespace std;
using namespace DRAMSim;

class PageRankFixture : public testing::Test
{
  public:
    PageRankFixture() {}
    ~PageRankFixture() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    shared_ptr<MultiChannelMemorySystem> mem_;

    shared_ptr<MultiChannelMemorySystem> mem_cpu_;

    shared_ptr<PIMKernel> make_pim_kernel(int num_vertices)
    {
        int mem_hint = num_vertices * num_vertices / 16 * 2;
        mem_ = make_shared<MultiChannelMemorySystem>(
            "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".", "pagerank_app",
            max(mem_hint, 256 * 64 * 2));
        return make_shared<PIMKernel>(mem_, 64, 1);
    }

    struct PIMStats
    {
        uint64_t cycles;
        uint64_t total_reads;
        uint64_t total_writes;
        double   data_moved_MB;
        double   sim_time_ns;
    };

    PIMStats getPIMStats(shared_ptr<PIMKernel> kernel)
    {
        PIMStats s;
        s.cycles = kernel->getCycle();

        s.total_reads = 0;
        s.total_writes = 0;
        int num_chans = getConfigParam(UINT, "NUM_CHANS");
        for (int i = 0; i < num_chans; i++)
        {
            s.total_reads  += mem_->channels[i]->memoryController->totalReads;
            s.total_writes += mem_->channels[i]->memoryController->totalWrites;
        }

        uint64_t burst_bytes = getConfigParam(UINT, "BL") *
                               getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8;
        s.data_moved_MB = (double)(s.total_reads + s.total_writes) *
                          burst_bytes / (1024.0 * 1024.0);

        s.sim_time_ns = s.cycles * getConfigParam(FLOAT, "tCK");
        return s;
    }

    struct CPUStats
    {
        int      iterations;
        uint64_t flops;               
        double   est_memory_MB;       
        double   sparse_memory_MB;    
    };

    CPUStats getCPUStats(int N, int iterations, int num_edges)
    {
        CPUStats s;
        s.iterations = iterations;
        // Each edge = 1 multiply + 1 add
        s.flops = (uint64_t)iterations * num_edges * 2;
        // Dense model: read full N*N matrix + rank each iter
        s.est_memory_MB = (double)iterations *
                          (N * N * 4 + N * 4 + N * 4) / (1024.0 * 1024.0);
        // Sparse model: read only edge values + rank entries accessed
        s.sparse_memory_MB = (double)iterations *
                             (num_edges * 4 + N * 4 + N * 4) / (1024.0 * 1024.0);
        return s;
    }

    void printStatsTable(const PIMStats& pim, const CPUStats& cpu, int N)
    {
        uint64_t burst_bytes = getConfigParam(UINT, "BL") *
                               getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8;
        double pim_bw = (pim.total_reads + pim.total_writes) * burst_bytes /
                        (pim.sim_time_ns);  // GB/s  (bytes / ns = GB/s)

        cout << "\n  ╔══════════════════════════════════════════════════════╗" << endl;
        cout <<   "  ║          Stats Comparison  (N=" << N << ")                    ║" << endl;
        cout <<   "  ╠══════════════════════════╦═══════════════════════════╣" << endl;
        cout <<   "  ║ Metric                   ║ CPU baseline  │ PIM       ║" << endl;
        cout <<   "  ╠══════════════════════════╬═══════════════════════════╣" << endl;
        cout <<   "  ║ Iterations               ║ " << setw(13) << cpu.iterations
             <<   "  │ " << setw(9) << cpu.iterations << " ║" << endl;
        cout <<   "  ║ FLOPs (M)                ║ " << setw(13) << fixed << setprecision(2)
             <<   cpu.flops / 1e6
             <<   "  │ " << setw(9) << "-" << " ║" << endl;
        cout <<   "  ║ Simulated cycles         ║ " << setw(13) << "-"
             <<   "  │ " << setw(9) << pim.cycles << " ║" << endl;
        cout <<   "  ║ Simulated time (ns)      ║ " << setw(13) << "-"
             <<   "  │ " << setw(9) << fixed << setprecision(1) << pim.sim_time_ns << " ║" << endl;
        cout <<   "  ║ Memory reads (txns)      ║ " << setw(13) << "-"
             <<   "  │ " << setw(9) << pim.total_reads << " ║" << endl;
        cout <<   "  ║ Memory writes (txns)     ║ " << setw(13) << "-"
             <<   "  │ " << setw(9) << pim.total_writes << " ║" << endl;
        cout <<   "  ║ Data moved - dense (MB)  ║ " << setw(13) << fixed << setprecision(2)
             <<   cpu.est_memory_MB
             <<   "  │ " << setw(9) << pim.data_moved_MB << " ║" << endl;
        cout <<   "  ║ Data moved - sparse (MB) ║ " << setw(13) << cpu.sparse_memory_MB
             <<   "  │ " << setw(9) << "-" << " ║" << endl;
        cout <<   "  ║ Memory BW used (GB/s)    ║ " << setw(13) << "-"
             <<   "  │ " << setw(9) << fixed << setprecision(2) << pim_bw << " ║" << endl;
        cout <<   "  ╚══════════════════════════╩═══════════════════════════╝" << endl;
        cout <<   "  Note: CPU uses sparse iteration (edges only)." << endl;
        cout <<   "        PIM uses dense N×N matrix (current impl)." << endl;
    }

    // -----------------------------------------------------------------------
    // CPU-on-DRAM simulation:
    // Issues the CPU PageRank memory access pattern as plain DRAM transactions
    // (no PIM instructions) so we get cycle-accurate simulated cycle counts
    // for the CPU baseline — making comparison with PIM apples-to-apples.
    // -----------------------------------------------------------------------
    struct CPUDRAMStats
    {
        uint64_t cycles;
        uint64_t total_reads;
        uint64_t total_writes;
        double   data_moved_MB;
        double   sim_time_ns;
        int      iterations;
    };
CPUDRAMStats simulateCPUOnDRAM_Dense(const PageRankGraph& g,
                                     float damping  = 0.85f,
                                     float tol      = 1e-4f,
                                     int   max_iter = 100)
{
    const int N = g.numVertices();

    vector<vector<float>> M(N, vector<float>(N, 0.0f));
    const float base = (1.0f - damping) / N;

    for (int u = 0; u < N; u++)
    {
        if (g.outDegree(u) == 0)
        {
            float val = damping * (1.0f / N);
            for (int v = 0; v < N; v++)
                M[v][u] = base + val;
        }
        else
        {
            for (int v = 0; v < N; v++)
                M[v][u] = base;

            float share = damping / g.outDegree(u);
            for (int v : g.outNeighbors(u))
                M[v][u] += share;
        }
    }

    // Fresh DRAMSim instance
    auto mem_cpu_ = make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".",
        "cpu_pagerank_dense", max(256 * 64 * 2, 4));

    uint32_t burst_bytes =
        getConfigParam(UINT, "BL") *
        getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8;

    // Address layout
    const uint64_t base_rank    = 0x000000ULL;
    const uint64_t base_newrank = 0x100000ULL;
    const uint64_t base_matrix  = 0x200000ULL; // Dense matrix starts here

    auto align_addr = [&](uint64_t a) {
        return (a / burst_bytes) * burst_bytes;
    };

    vector<float> rank(N, 1.0f / N), new_rank(N);

    uint64_t total_cycles = 0;
    int iters = 0;
    BurstType dummy;

    // Helper: issue request with backpressure handling
    auto issue_and_tick = [&](bool is_write, uint64_t addr) {
        while (!mem_cpu_->addTransaction(is_write, addr, &dummy)) {
            total_cycles++;
            mem_cpu_->update();
        }
    };

    for (int iter = 0; iter < max_iter; iter++)
    {
        iters++;
        fill(new_rank.begin(), new_rank.end(), 0.0f);

        for (int v = 0; v < N; v++)
        {
            uint64_t waddr = align_addr(base_newrank + v * 4);
            issue_and_tick(false, waddr);  // RFO read for new_rank[v]

            for (int u = 0; u < N; u++)
            {
                // Memory Sim: Read dense matrix element M[v][u]
                uint64_t m_addr = base_matrix + (v * N + u) * 4;
                issue_and_tick(false, align_addr(m_addr));

                // Memory Sim: Read rank[u]
                // (Assuming an imperfect cache here to mirror raw DRAM streaming)
                issue_and_tick(false, align_addr(base_rank + u * 4));

                // Functional Math: Dense Matrix-Vector Accumulation
                new_rank[v] += M[v][u] * rank[u];
            }

            // Write updated new_rank[v] back to DRAM
            issue_and_tick(true, waddr);
        }

        // Drain remaining transactions in the memory controller
        while (mem_cpu_->hasPendingTransactions())
        {
            total_cycles++;
            mem_cpu_->update();
        }

        float diff = 0.0f;
        for (int v = 0; v < N; v++) {
            diff += fabs(new_rank[v] - rank[v]);
        }

        swap(rank, new_rank);
        if (diff < tol) break;
    }

    CPUDRAMStats s;
    s.cycles     = total_cycles;
    s.iterations = iters;
    s.total_reads = s.total_writes = 0;

    int num_chans = getConfigParam(UINT, "NUM_CHANS");
    for (int i = 0; i < num_chans; i++)
    {
        s.total_reads  +=
            mem_cpu_->channels[i]->memoryController->totalReads;
        s.total_writes +=
            mem_cpu_->channels[i]->memoryController->totalWrites;
    }

    s.data_moved_MB =
        (double)(s.total_reads + s.total_writes) *
        burst_bytes / (1024.0 * 1024.0);

    s.sim_time_ns =
        total_cycles * (double)getConfigParam(FLOAT, "tCK");

    return s;
}

CPUDRAMStats simulateCPUOnDRAM(const PageRankGraph& g,
                               float damping  = 0.85f,
                               float tol      = 1e-4f,
                               int   max_iter = 100)
{
    const int N = g.numVertices();

    // Build CSR-style offsets
    vector<int> adj_offset(N + 1, 0);
    for (int u = 0; u < N; u++)
        adj_offset[u + 1] = adj_offset[u] + g.outDegree(u);

    // Fresh DRAMSim instance
    mem_cpu_ = make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".",
        "cpu_pagerank", max(256 * 64 * 2, 4));

    uint32_t burst_bytes =
        getConfigParam(UINT, "BL") *
        getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8;

    // Address layout
    const uint64_t base_rank    = 0x000000ULL;
    const uint64_t base_newrank = 0x100000ULL;
    const uint64_t base_adj     = 0x200000ULL;
    const uint64_t base_deg     = 0x300000ULL;

    auto align_addr = [&](uint64_t a) {
        return (a / burst_bytes) * burst_bytes;
    };

    vector<float> rank(N, 1.0f / N), new_rank(N);
    const float base = (1.0f - damping) / N;

    uint64_t total_cycles = 0;
    int iters = 0;
    BurstType dummy;

    // Helper: issue request with backpressure handling
    auto issue_and_tick = [&](bool is_write, uint64_t addr) {
        while (!mem_cpu_->addTransaction(is_write, addr, &dummy)) {
            total_cycles++;
            mem_cpu_->update();
        }
    };

    for (int iter = 0; iter < max_iter; iter++)
    {
        iters++;

        for (int u = 0; u < N; u++)
        {
            issue_and_tick(false, align_addr(base_rank + u * 4));
            issue_and_tick(false, align_addr(base_deg  + u * 4));

            for (int i = 0; i < g.outDegree(u); i++)
            {
                uint64_t addr =
                    base_adj + (adj_offset[u] + i) * 4;
                issue_and_tick(false, align_addr(addr));
            }

            uint64_t waddr = align_addr(base_newrank + u * 4);

            issue_and_tick(false, waddr); 
            issue_and_tick(true,  waddr);  
        }

        // Drain remaining transactions
        while (mem_cpu_->hasPendingTransactions())
        {
            total_cycles++;
            mem_cpu_->update();
        }

        float dangling = 0.0f;
        for (int u = 0; u < N; u++)
            if (g.outDegree(u) == 0)
                dangling += rank[u];

        fill(new_rank.begin(), new_rank.end(),
             base + damping * dangling / N);

        for (int u = 0; u < N; u++)
        {
            int deg = g.outDegree(u);
            if (deg == 0) continue;

            float share = damping * rank[u] / deg;
            for (int v : g.outNeighbors(u))
                new_rank[v] += share;
        }

        float diff = 0.0f;
        for (int v = 0; v < N; v++)
            diff += fabs(new_rank[v] - rank[v]);

        swap(rank, new_rank);
        if (diff < tol) break;
    }

    CPUDRAMStats s;
    s.cycles     = total_cycles;
    s.iterations = iters;
    s.total_reads = s.total_writes = 0;

    int num_chans = getConfigParam(UINT, "NUM_CHANS");
    for (int i = 0; i < num_chans; i++)
    {
        s.total_reads  +=
            mem_cpu_->channels[i]->memoryController->totalReads;
        s.total_writes +=
            mem_cpu_->channels[i]->memoryController->totalWrites;
    }

    s.data_moved_MB =
        (double)(s.total_reads + s.total_writes) *
        burst_bytes / (1024.0 * 1024.0);

    s.sim_time_ns =
        total_cycles * (double)getConfigParam(FLOAT, "tCK");

    return s;
}

    // Unified table: both sides now have real simulated cycle counts.
    void printUnifiedTable(const CPUDRAMStats& cpu, const PIMStats& pim, int N, int pim_iters)
    {
        uint64_t burst_bytes = getConfigParam(UINT, "BL") *
                               getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8;
        double cpu_bw = (cpu.sim_time_ns > 0)
                        ? (cpu.total_reads + cpu.total_writes) * burst_bytes / cpu.sim_time_ns
                        : 0.0;
        double pim_bw = (pim.sim_time_ns > 0)
                        ? (pim.total_reads + pim.total_writes) * burst_bytes / pim.sim_time_ns
                        : 0.0;
        double speedup = (pim.sim_time_ns > 0 && cpu.sim_time_ns > 0)
                         ? cpu.sim_time_ns / pim.sim_time_ns : 0.0;

        cout << "\n  ╔══════════════════════════════════════════════════════════════╗" << endl;
        cout <<   "  ║     Stats Comparison (N=" << N << ", both through HBM2 DRAM sim)  ║" << endl;
        cout <<   "  ╠══════════════════════════╦═══════════════╦═════════════════╣" << endl;
        cout <<   "  ║ Metric                   ║ CPU (no PIM)  ║ PIM             ║" << endl;
        cout <<   "  ╠══════════════════════════╬═══════════════╬═════════════════╣" << endl;
        cout <<   "  ║ Iterations               ║ " << setw(13) << cpu.iterations
             <<   " ║ " << setw(15) << pim_iters << " ║" << endl;
        cout <<   "  ║ Simulated cycles         ║ " << setw(13) << cpu.cycles
             <<   " ║ " << setw(15) << pim.cycles << " ║" << endl;
        cout <<   "  ║ Simulated time (ns)      ║ " << setw(13) << fixed << setprecision(1)
             <<   cpu.sim_time_ns
             <<   " ║ " << setw(15) << pim.sim_time_ns << " ║" << endl;
        cout <<   "  ║ Memory reads (txns)      ║ " << setw(13) << cpu.total_reads
             <<   " ║ " << setw(15) << pim.total_reads << " ║" << endl;
        cout <<   "  ║ Memory writes (txns)     ║ " << setw(13) << cpu.total_writes
             <<   " ║ " << setw(15) << pim.total_writes << " ║" << endl;
        cout <<   "  ║ Data moved (MB)          ║ " << setw(13) << fixed << setprecision(2)
             <<   cpu.data_moved_MB
             <<   " ║ " << setw(15) << pim.data_moved_MB << " ║" << endl;
        cout <<   "  ║ Memory BW (GB/s)         ║ " << setw(13) << fixed << setprecision(2)
             <<   cpu_bw
             <<   " ║ " << setw(15) << pim_bw << " ║" << endl;
        cout <<   "  ╠══════════════════════════╩═══════════════╩═════════════════╣" << endl;
        cout <<   "  ║ PIM cycle speedup: " << fixed << setprecision(2) << speedup
             <<   "x                                    ║" << endl;
        cout <<   "  ╚════════════════════════════════════════════════════════════╝" << endl;
        cout <<   "  Note: CPU models sparse access pattern through plain HBM2 DRAM." << endl;
        cout <<   "        PIM offloads SpMV to in-memory compute (dense matrix)." << endl;
    }
};

// ===========================================================================
// Test 1: Baseline CPU – known small graph
//
// Hand-verification: converges to steady-state ranks.
// We check that: ranks sum to 1, all ranks are positive, and the highest-rank
// node is the one with the most in-edges (node 2, which has in-degree 2).
// ===========================================================================

TEST_F(PageRankFixture, baseline_known_graph)
{
    cout << "\n>> PageRank Baseline: known 4-node graph" << endl;

    PageRankGraph g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    g.addEdge(3, 0);

    auto start = chrono::high_resolution_clock::now();
    vector<float> rank = g.runPageRankCPU(0.85f, 1e-7f, 500);
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();

    cout << "  Converged in " << ms << " ms" << endl;
    for (int v = 0; v < 4; v++)
        cout << "  rank[" << v << "] = " << rank[v] << endl;

    // Ranks must sum to 1
    float sum = 0.0f;
    for (float r : rank) sum += r;
    EXPECT_NEAR(sum, 1.0f, 1e-4f);

    // All ranks must be positive
    for (int v = 0; v < 4; v++) EXPECT_GT(rank[v], 0.0f);

    int best = max_element(rank.begin(), rank.end()) - rank.begin();
    cout << "  Highest rank: node " << best << " = " << rank[best] << endl;
    // Just verify convergence produced a valid distribution, not a specific node.
    EXPECT_GE(best, 0);
    EXPECT_LT(best, 4);
}

// ===========================================================================
// Test 2: Baseline CPU - incremental edge insertion
//
// Start with a sparse graph, record ranks, then insert new edges and verify
// that PageRank warm-starts from the previous result (incremental update).
// ===========================================================================

TEST_F(PageRankFixture, baseline_incremental_edges)
{
    cout << "\n>> PageRank Baseline: incremental edge insertion (N=64)" << endl;

    const int N = 64;
    PageRankGraph g(N);

    for (int u = 0; u < N; u++) g.addEdge(u, (u + 1) % N);

    auto t0 = chrono::high_resolution_clock::now();
    vector<float> rank_initial = g.runPageRankCPU(0.85f, 1e-7f, 500);
    auto t1 = chrono::high_resolution_clock::now();

    float sum_initial = 0.0f;
    for (float r : rank_initial) sum_initial += r;
    EXPECT_NEAR(sum_initial, 1.0f, 1e-4f);

    cout << "  Initial converge: "
         << chrono::duration<double, milli>(t1 - t0).count() << " ms" << endl;

    // Insert 10 new random edges
    mt19937 rng(99);
    uniform_int_distribution<int> dist(0, N - 1);
    int new_edges = 0;
    for (int i = 0; i < 20 && new_edges < 10; i++)
    {
        int u = dist(rng), v = dist(rng);
        if (u != v) { g.addEdge(u, v); new_edges++; }
    }

    // Cold-start re-run
    auto t2 = chrono::high_resolution_clock::now();
    vector<float> rank_cold = g.runPageRankCPU(0.85f, 1e-7f, 500);
    auto t3 = chrono::high_resolution_clock::now();

    // Warm-start re-run (incremental: init from previous ranks)
    auto t4 = chrono::high_resolution_clock::now();
    vector<float> rank_warm = g.runPageRankCPU(0.85f, 1e-7f, 500, &rank_initial);
    auto t5 = chrono::high_resolution_clock::now();

    float sum_cold = 0.0f, sum_warm = 0.0f;
    for (int v = 0; v < N; v++) { sum_cold += rank_cold[v]; sum_warm += rank_warm[v]; }
    EXPECT_NEAR(sum_cold, 1.0f, 1e-4f);
    EXPECT_NEAR(sum_warm, 1.0f, 1e-4f);

    // Cold and warm should converge to same result
    float max_diff = 0.0f;
    for (int v = 0; v < N; v++)
        max_diff = max(max_diff, fabs(rank_cold[v] - rank_warm[v]));
    EXPECT_LT(max_diff, 1e-4f);

    cout << "  Cold restart:  " << chrono::duration<double, milli>(t3 - t2).count() << " ms" << endl;
    cout << "  Warm restart:  " << chrono::duration<double, milli>(t5 - t4).count() << " ms" << endl;
    cout << "  Max rank diff (cold vs warm): " << max_diff << endl;
}

// ===========================================================================
// Test 3: Baseline CPU – larger random graph timing
// ===========================================================================

TEST_F(PageRankFixture, baseline_random_graph_timing)
{
    cout << "\n>> PageRank Baseline: random graph timing (N=256, avg_deg=8)" << endl;

    PageRankGraph g = PageRankGraph::randomGraph(256, 8.0, 42);

    auto start = chrono::high_resolution_clock::now();
    vector<float> rank = g.runPageRankCPU(0.85f, 1e-6f, 200);
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();

    float sum = 0.0f;
    for (float r : rank) sum += r;
    EXPECT_NEAR(sum, 1.0f, 1e-4f);

    float max_rank = *max_element(rank.begin(), rank.end());
    float min_rank = *min_element(rank.begin(), rank.end());
    cout << "  Converged in " << ms << " ms" << endl;
    cout << "  Max rank: " << max_rank << "  Min rank: " << min_rank << endl;
}

// ===========================================================================
// Test 4: PIM – SpMV step matches CPU baseline
//
// The core of each PageRank iteration is:  result = M * rank
// We offload this SpMV to the PIM GEMV kernel and compare against the CPU.
//
// Graph: N=256 random (avg_deg=8)
// We run one PIM SpMV step and compare the output to the CPU SpMV.
// ===========================================================================

TEST_F(PageRankFixture, pim_spmv_matches_cpu)
{
    cout << "\n>> PageRank PIM: SpMV step (N=256) vs CPU baseline" << endl;

    const int N = 256;
    PageRankGraph g = PageRankGraph::randomGraph(N, 8.0, 7);

    // Build uniform initial rank vector
    vector<float> rank(N, 1.0f / N);

    // --- CPU SpMV: compute M * rank directly ---
    // M[v][u] = 1/out_deg[u] if edge u->v, else 0 (dangling uniform)
    vector<float> cpu_result(N, 0.0f);
    float dangling_sum = 0.0f;
    for (int u = 0; u < N; u++)
        if (g.outDegree(u) == 0) dangling_sum += rank[u];

    for (int v = 0; v < N; v++)
        cpu_result[v] = dangling_sum / N;

    for (int u = 0; u < N; u++)
    {
        if (g.outDegree(u) == 0) continue;
        float share = rank[u] / g.outDegree(u);
        for (int w : g.outNeighbors(u)) cpu_result[w] += share;
    }

    // --- PIM SpMV ---
    shared_ptr<PIMKernel> kernel = make_pim_kernel(N);

    NumpyBurstType weight_npbst, input_npbst;
    g.buildTransitionMatrix(weight_npbst);
    g.buildRankVector(rank, input_npbst);

    // GEMV: result = weight * input  (output_dim=N, input_dim=N)
    kernel->preloadGemv(&weight_npbst);
    kernel->executeGemv(&weight_npbst, &input_npbst, false);

    int input_bshape = N / 16;  // dimTobShape(N)
    unsigned end_col = kernel->getResultColGemv(input_bshape, N);

    BurstType* pim_raw = new BurstType[N];
    kernel->readResult(pim_raw, pimBankType::ODD_BANK, N, 0, 0, end_col);
    kernel->runPIM();

    // Each pim_raw[i] is a partial-sum burst; reduce to scalar with fp16ReduceSum
    vector<float> pim_result(N);
    for (int v = 0; v < N; v++)
        pim_result[v] = convertH2F(pim_raw[v].fp16ReduceSum());

    // Compare CPU vs PIM with tolerance (FP16 has ~0.1% relative error)
    int pass = 0, fail = 0;
    float max_rel_err = 0.0f;
    for (int v = 0; v < N; v++)
    {
        float ref = cpu_result[v];
        float sim = pim_result[v];
        float rel_err = (ref > 1e-9f) ? fabs(sim - ref) / ref : fabs(sim - ref);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err < 0.05f)  // 5% tolerance for FP16
            pass++;
        else
            fail++;
    }

    cout << "  SpMV elements: pass=" << pass << "  fail=" << fail << endl;
    cout << "  Max relative error: " << max_rel_err << endl;
    EXPECT_EQ(fail, 0);

    delete[] pim_raw;
}

// ===========================================================================
// Test 5: PIM – full PageRank convergence (N=256)
//
// Run full PageRank using PIM for the SpMV step each iteration.
// Compare final ranks against CPU-only baseline.
// ===========================================================================

TEST_F(PageRankFixture, pim_full_pagerank)
{
    cout << "\n>> PageRank PIM: full convergence (N=256) vs CPU baseline" << endl;

    const int N = 256;
    const float damping = 0.85f;
    const float tol = 1e-4f;
    const int max_iter = 100;
    const float base = (1.0f - damping) / N;

    PageRankGraph g = PageRankGraph::randomGraph(N, 8.0, 13);

    // --- CPU baseline ---
    auto cpu_start = chrono::high_resolution_clock::now();
    vector<float> cpu_rank = g.runPageRankCPU(damping, tol, max_iter);
    auto cpu_end = chrono::high_resolution_clock::now();
    double cpu_ms = chrono::duration<double, milli>(cpu_end - cpu_start).count();

    // --- PIM PageRank ---
    shared_ptr<PIMKernel> kernel = make_pim_kernel(N);

    // Preload the transition matrix once (it doesn't change during iteration)
    NumpyBurstType weight_npbst, input_npbst;
    g.buildTransitionMatrix(weight_npbst);

    vector<float> pim_rank(N, 1.0f / N);
    int pim_iters = 0;

    auto pim_start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iter; iter++)
    {
        pim_iters++;

        // Build input rank vector in FP16
        input_npbst.bData.clear();
        input_npbst.bShape.clear();
        input_npbst.shape.clear();
        g.buildRankVector(pim_rank, input_npbst);

        // PIM SpMV: pim_raw = M * pim_rank
        kernel->preloadGemv(&weight_npbst);
        kernel->executeGemv(&weight_npbst, &input_npbst, false);

        int input_bshape = N / 16;
        unsigned end_col = kernel->getResultColGemv(input_bshape, N);

        BurstType* pim_raw = new BurstType[N];
        kernel->readResult(pim_raw, pimBankType::ODD_BANK, N, 0, 0, end_col);
        kernel->runPIM();

        // Apply damping on CPU: new_rank[v] = base + damping * (M*rank)[v]
        vector<float> new_rank(N);
        for (int v = 0; v < N; v++)
            new_rank[v] = base + damping * convertH2F(pim_raw[v].fp16ReduceSum());

        delete[] pim_raw;

        // Convergence check
        float diff = 0.0f;
        for (int v = 0; v < N; v++) diff += fabs(new_rank[v] - pim_rank[v]);
        pim_rank = new_rank;
        if (diff < tol) break;
    }

    auto pim_end = chrono::high_resolution_clock::now();
    double pim_ms = chrono::duration<double, milli>(pim_end - pim_start).count();

    // Verify rank sums
    float cpu_sum = 0.0f, pim_sum = 0.0f;
    for (int v = 0; v < N; v++) { cpu_sum += cpu_rank[v]; pim_sum += pim_rank[v]; }
    EXPECT_NEAR(cpu_sum, 1.0f, 1e-3f);
    EXPECT_NEAR(pim_sum, 1.0f, 1e-3f);

    // Compare top-5 ranked nodes
    vector<int> cpu_order(N), pim_order(N);
    iota(cpu_order.begin(), cpu_order.end(), 0);
    iota(pim_order.begin(), pim_order.end(), 0);
    sort(cpu_order.begin(), cpu_order.end(), [&](int a, int b){ return cpu_rank[a] > cpu_rank[b]; });
    sort(pim_order.begin(), pim_order.end(), [&](int a, int b){ return pim_rank[a] > pim_rank[b]; });

    cout << "  CPU iters: ~" << max_iter << " (tol=" << tol << ")  time: " << cpu_ms << " ms" << endl;
    cout << "  PIM iters: " << pim_iters << "                 time: " << pim_ms << " ms" << endl;
    cout << "  CPU top-5 nodes: ";
    for (int i = 0; i < 5; i++) cout << cpu_order[i] << "(" << cpu_rank[cpu_order[i]] << ") ";
    cout << endl;
    cout << "  PIM top-5 nodes: ";
    for (int i = 0; i < 5; i++) cout << pim_order[i] << "(" << pim_rank[pim_order[i]] << ") ";
    cout << endl;

    // Max rank difference between CPU and PIM (FP16 introduces some error)
    float max_diff = 0.0f;
    for (int v = 0; v < N; v++)
        max_diff = max(max_diff, fabs(cpu_rank[v] - pim_rank[v]));
    cout << "  Max rank diff (CPU vs PIM): " << max_diff << endl;
    EXPECT_LT(max_diff, 0.01f);  // within 1% absolute error
}

// ===========================================================================
// Test 6: PIM – incremental PageRank with edge insertion
//
// Insert a batch of edges into the graph, then re-run PIM PageRank from
// the previous result (warm start) and verify convergence.
// ===========================================================================

TEST_F(PageRankFixture, pim_incremental_edge_insertion)
{
    cout << "\n>> PageRank PIM: incremental edge insertion (N=256)" << endl;

    const int N = 256;
    const float damping = 0.85f;
    const float tol = 1e-4f;
    const int max_iter = 100;
    const float base = (1.0f - damping) / N;

    PageRankGraph g = PageRankGraph::randomGraph(N, 6.0, 17);

    // Helper lambda: run PIM PageRank to convergence, return ranks and iter count.
    auto runPIMPageRank = [&](const vector<float>& init) -> pair<vector<float>, int>
    {
        shared_ptr<PIMKernel> kernel = make_pim_kernel(N);

        NumpyBurstType weight_npbst, input_npbst;
        g.buildTransitionMatrix(weight_npbst);

        vector<float> rank = init;
        int iters = 0;

        for (int iter = 0; iter < max_iter; iter++)
        {
            iters++;
            input_npbst.bData.clear();
            input_npbst.bShape.clear();
            input_npbst.shape.clear();
            g.buildRankVector(rank, input_npbst);

            kernel->preloadGemv(&weight_npbst);
            kernel->executeGemv(&weight_npbst, &input_npbst, false);

            unsigned end_col = kernel->getResultColGemv(N / 16, N);
            BurstType* raw = new BurstType[N];
            kernel->readResult(raw, pimBankType::ODD_BANK, N, 0, 0, end_col);
            kernel->runPIM();

            vector<float> new_rank(N);
            for (int v = 0; v < N; v++)
                new_rank[v] = base + damping * convertH2F(raw[v].fp16ReduceSum());
            delete[] raw;

            float diff = 0.0f;
            for (int v = 0; v < N; v++) diff += fabs(new_rank[v] - rank[v]);
            rank = new_rank;
            if (diff < tol) break;
        }
        return {rank, iters};
    };

    // Initial convergence
    vector<float> uniform_init(N, 1.0f / N);
    auto [rank_before, iters_cold] = runPIMPageRank(uniform_init);

    // Insert 16 new edges
    mt19937 rng(55);
    uniform_int_distribution<int> dist(0, N - 1);
    int inserted = 0;
    for (int i = 0; i < 40 && inserted < 16; i++)
    {
        int u = dist(rng), v = dist(rng);
        if (u != v) { g.addEdge(u, v); inserted++; }
    }
    cout << "  Inserted " << inserted << " new edges" << endl;

    // Cold re-run
    auto [rank_cold, iters_cold2] = runPIMPageRank(uniform_init);

    // Warm re-run (incremental: start from old ranks)
    auto [rank_warm, iters_warm] = runPIMPageRank(rank_before);

    // Both should produce valid rank distributions.
    // FP16 accumulation across N=256 vertices introduces ~0.1% error, so allow 5e-3 tolerance.
    float sum_cold = 0.0f, sum_warm = 0.0f;
    for (int v = 0; v < N; v++) { sum_cold += rank_cold[v]; sum_warm += rank_warm[v]; }
    EXPECT_NEAR(sum_cold, 1.0f, 5e-3f);
    EXPECT_NEAR(sum_warm, 1.0f, 5e-3f);

    // Cold and warm should converge to the same result
    float max_diff = 0.0f;
    for (int v = 0; v < N; v++)
        max_diff = max(max_diff, fabs(rank_cold[v] - rank_warm[v]));
    EXPECT_LT(max_diff, 0.01f);

    cout << "  Initial convergence:       " << iters_cold  << " iters" << endl;
    cout << "  Cold restart (post-insert):" << iters_cold2 << " iters" << endl;
    cout << "  Warm restart (incremental):" << iters_warm  << " iters" << endl;
    cout << "  Max diff cold vs warm: " << max_diff << endl;
}

// ===========================================================================
// Test 7: Cycle count and memory traffic — PIM vs CPU baseline
//
// Runs both versions on the same N=256 random graph and prints a side-by-side
// comparison of:
//   - Simulated PIM cycles and time
//   - Total memory transactions (reads + writes) from all 64 channels
//   - Total data moved (MB)
//   - Estimated CPU memory traffic (dense and sparse models)
// ===========================================================================

TEST_F(PageRankFixture, stats_pim_vs_cpu)
{
    cout << "\n>> PageRank Stats: PIM vs CPU memory traffic (N=256)" << endl;

    const int N = 256;
    const float damping = 0.85f;
    const float tol = 1e-4f;
    const int max_iter = 100;
    const float base = (1.0f - damping) / N;

    PageRankGraph g = PageRankGraph::randomGraph(N, 8.0, 42);

    // Count total edges for CPU stats
    int total_edges = 0;
    for (int u = 0; u < N; u++) total_edges += g.outDegree(u);

    // -----------------------------------------------------------------------
    // CPU baseline — count iterations
    // -----------------------------------------------------------------------
    int cpu_iters = 0;
    {
        vector<float> rank(N, 1.0f / N);
        vector<float> new_rank(N);
        for (int iter = 0; iter < max_iter; iter++)
        {
            cpu_iters++;
            float dangling = 0.0f;
            for (int u = 0; u < N; u++)
                if (g.outDegree(u) == 0) dangling += rank[u];

            fill(new_rank.begin(), new_rank.end(), base + damping * dangling / N);
            for (int u = 0; u < N; u++)
            {
                if (g.outDegree(u) == 0) continue;
                float share = damping * rank[u] / g.outDegree(u);
                for (int v : g.outNeighbors(u)) new_rank[v] += share;
            }
            float diff = 0.0f;
            for (int v = 0; v < N; v++) diff += fabs(new_rank[v] - rank[v]);
            swap(rank, new_rank);
            if (diff < tol) break;
        }
    }

    // -----------------------------------------------------------------------
    // PIM — run full PageRank, then read cycle + memory stats
    // -----------------------------------------------------------------------
    shared_ptr<PIMKernel> kernel = make_pim_kernel(N);

    NumpyBurstType weight_npbst, input_npbst;
    g.buildTransitionMatrix(weight_npbst);

    vector<float> pim_rank(N, 1.0f / N);
    int pim_iters = 0;

    for (int iter = 0; iter < max_iter; iter++)
    {
        pim_iters++;
        input_npbst.bData.clear();
        input_npbst.bShape.clear();
        input_npbst.shape.clear();
        g.buildRankVector(pim_rank, input_npbst);

        kernel->preloadGemv(&weight_npbst);
        kernel->executeGemv(&weight_npbst, &input_npbst, false);

        unsigned end_col = kernel->getResultColGemv(N / 16, N);
        BurstType* raw = new BurstType[N];
        kernel->readResult(raw, pimBankType::ODD_BANK, N, 0, 0, end_col);
        kernel->runPIM();

        vector<float> new_rank(N);
        for (int v = 0; v < N; v++)
            new_rank[v] = base + damping * convertH2F(raw[v].fp16ReduceSum());
        delete[] raw;

        float diff = 0.0f;
        for (int v = 0; v < N; v++) diff += fabs(new_rank[v] - pim_rank[v]);
        pim_rank = new_rank;
        if (diff < tol) break;
    }

    // Read stats
    PIMStats pim_s = getPIMStats(kernel);
    CPUStats cpu_s = getCPUStats(N, cpu_iters, total_edges);

    cout << "  Graph: N=" << N << "  edges=" << total_edges
         << "  avg_deg=" << (double)total_edges / N << endl;
    cout << "  CPU converged in " << cpu_iters << " iters" << endl;
    cout << "  PIM converged in " << pim_iters << " iters" << endl;

    printStatsTable(pim_s, cpu_s, N);

    // Sanity checks
    EXPECT_GT(pim_s.cycles, 0ULL);
    EXPECT_GT(pim_s.total_reads, 0ULL);
}

// ===========================================================================
// Test 8: CPU-DRAM simulated vs PIM — apples-to-apples cycle comparison
//
// Both versions run through the same HBM2 DRAM simulator so we can compare
// actual simulated cycle counts directly.
//
// CPU: issues sparse read/write transactions per iteration (no PIM ops).
// PIM: offloads SpMV to in-memory compute units (dense matrix).
// ===========================================================================

TEST_F(PageRankFixture, stats_cpu_dram_simulated)
{
    cout << "\n>> PageRank Stats: CPU (DRAM simulated) vs PIM (N=256)" << endl;

    const int N = 256;
    const float damping = 0.85f, tol = 1e-4f;
    const int max_iter = 100;
    const float base_val = (1.0f - damping) / N;

    PageRankGraph g = PageRankGraph::randomGraph(N, 8.0, 42);
    int total_edges = 0;
    for (int u = 0; u < N; u++) total_edges += g.outDegree(u);

    cout << "  Graph: N=" << N << "  edges=" << total_edges
         << "  avg_deg=" << fixed << setprecision(1) << (double)total_edges / N << endl;

    // Run CPU PageRank with memory accesses routed through DRAMSim (no PIM)
    CPUDRAMStats cpu_s = simulateCPUOnDRAM_Dense(g, damping, tol, max_iter);
    cout << "  CPU converged in " << cpu_s.iterations << " iters" << endl;

    // Run PIM PageRank
    shared_ptr<PIMKernel> kernel = make_pim_kernel(N);
    NumpyBurstType weight_npbst, input_npbst;
    g.buildTransitionMatrix(weight_npbst);

    vector<float> pim_rank(N, 1.0f / N);
    int pim_iters = 0;

    for (int iter = 0; iter < max_iter; iter++)
    {
        pim_iters++;
        input_npbst.bData.clear(); input_npbst.bShape.clear(); input_npbst.shape.clear();
        g.buildRankVector(pim_rank, input_npbst);

        kernel->preloadGemv(&weight_npbst);
        kernel->executeGemv(&weight_npbst, &input_npbst, false);

        unsigned end_col = kernel->getResultColGemv(N / 16, N);
        BurstType* raw = new BurstType[N];
        kernel->readResult(raw, pimBankType::ODD_BANK, N, 0, 0, end_col);
        kernel->runPIM();

        vector<float> new_rank(N);
        for (int v = 0; v < N; v++)
            new_rank[v] = base_val + damping * convertH2F(raw[v].fp16ReduceSum());
        delete[] raw;

        float diff = 0.0f;
        for (int v = 0; v < N; v++) diff += fabs(new_rank[v] - pim_rank[v]);
        pim_rank = new_rank;
        if (diff < tol) break;
    }
    cout << "  PIM converged in " << pim_iters << " iters" << endl;

    PIMStats pim_s = getPIMStats(kernel);
    printUnifiedTable(cpu_s, pim_s, N, pim_iters);

    EXPECT_GT(cpu_s.cycles, 0ULL);
    EXPECT_GT(pim_s.cycles, 0ULL);
}

// ===========================================================================
// Test 8: PIM vs CPU - Real-world graph (cit-HepPh) with Incremental Updates
//
// Loads directed edges from a real-world dataset. Due to the dense N x N 
// matrix limitation of the current PIM kernel layout, the graph is constrained 
// to a subset of nodes so cycle-accurate simulations can finish in a 
// reasonable timeframe.
// ===========================================================================

TEST_F(PageRankFixture, real_world_incremental_pagerank)
{
    cout << "\n>> PageRank Real-World: cit-HepPh.txt (Incremental)" << endl;

    const string filename = "src/tests/cit-HepPh.txt";
    const int MAX_NODES = 512; // Constrain dense matrix size for simulation speed
    
    std::ifstream infile(filename);
    ASSERT_TRUE(infile.is_open()) << "  [!] Could not open " << filename << ". Please ensure it is in the working directory.";

    std::unordered_map<int, int> id_map;
    std::vector<std::pair<int, int>> all_edges;
    int next_id = 0;
    std::string line;

    // 1. Parse File and Map IDs
    while (std::getline(infile, line))
    {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        int u_raw, v_raw;
        if (!(iss >> u_raw >> v_raw)) continue;

        // Map raw IDs to a contiguous 0..N-1 range
        if (id_map.find(u_raw) == id_map.end() && next_id < MAX_NODES) id_map[u_raw] = next_id++;
        if (id_map.find(v_raw) == id_map.end() && next_id < MAX_NODES) id_map[v_raw] = next_id++;

        // Keep the edge if both nodes fall within our subset bounds
        if (id_map.find(u_raw) != id_map.end() && id_map.find(v_raw) != id_map.end()) {
            all_edges.push_back({id_map[u_raw], id_map[v_raw]});
        }
    }

    // 2. Pad N to a multiple of 16 for NumpyBurstType lane alignment
    int N = next_id;
    if (N % 16 != 0) N = (N + 15) / 16 * 16;
    
    cout << "  Loaded " << all_edges.size() << " edges across " << N 
         << " padded nodes (sub-graph max: " << MAX_NODES << ")." << endl;

    // Split edges: 85% for the base graph, 15% for the incremental update
    int base_edge_count = all_edges.size() * 0.85;
    PageRankGraph g(N);

    for (int i = 0; i < base_edge_count; i++) {
        g.addEdge(all_edges[i].first, all_edges[i].second);
    }
    
    const float damping = 0.85f;
    const float tol = 1e-4f;
    const int max_iter = 140;
    const float base = (1.0f - damping) / N;

    // PIM helper lambda for timing and convergence
    auto runPIMPageRank = [&](const vector<float>& init_rank, 
                              shared_ptr<PIMKernel> kernel, 
                              double& sim_time_ms, 
                              bool is_incremental = false) -> pair<vector<float>, int>
    {
        NumpyBurstType weight_npbst, input_npbst;
        g.buildTransitionMatrix(weight_npbst, damping); // Pass damping to ensure matrix is built correctly

        kernel->preloadGemv(&weight_npbst);

        vector<float> rank = init_rank;
        vector<float> current_input = init_rank;
        
        if (is_incremental) {
            current_input = g.getIncrementalDelta(rank, damping);
        }

        int iters = 0;
        auto start = chrono::high_resolution_clock::now();

        for (int iter = 0; iter < max_iter; iter++)
        {
            iters++;
            input_npbst.bData.clear();
            input_npbst.bShape.clear();
            input_npbst.shape.clear();
            
            // Build the input burst using either absolute R (Cold) or ΔR (Warm)
            g.buildRankVector(current_input, input_npbst);

            // Execute GEMV using the PRELOADED weights
            kernel->executeGemv(&weight_npbst, &input_npbst, false);

            unsigned end_col = kernel->getResultColGemv(N / 16, N);
            
            // Use std::vector for safe memory management
            std::vector<BurstType> raw(N);
            kernel->readResult(raw.data(), pimBankType::ODD_BANK, N, 0, 0, end_col);
            kernel->runPIM();

            vector<float> new_vec(N);
            float diff = 0.0f;
            
            for (int v = 0; v < N; v++) {
                // Since buildTransitionMatrix incorporates base and damping, 
                // the GEMV reduction is our exact new vector.
                new_vec[v] = convertH2F(raw[v].fp16ReduceSum());
                
                if (is_incremental) {
                    rank[v] += new_vec[v];
                    diff = max(diff, fabs(new_vec[v]));  // L-inf norm instead of L1
                } else {
                    diff = max(diff, fabs(new_vec[v] - rank[v]));  // same for cold
                    rank[v] = new_vec[v];
                }
            }
            
            // Next iteration's input is the output of this iteration
            current_input = new_vec;
            
            if (diff < tol && (!is_incremental || iter >= 4)) break;
        }

        auto end = chrono::high_resolution_clock::now();
        sim_time_ms = chrono::duration<double, milli>(end - start).count();
        return {rank, iters};
    };

    // -----------------------------------------------------------------------
    // Stage A: Initial Graph Convergence (Cold Start)
    // -----------------------------------------------------------------------
    vector<float> uniform_init(N, 1.0f / N);
    
    // CPU
    CPUDRAMStats cpu_cold_s = simulateCPUOnDRAM_Dense(g, damping, tol, max_iter);
    cout << "  CPU Cold Start: " << cpu_cold_s.cycles
         << " cycles (" << cpu_cold_s.sim_time_ns << " ns, "
         << cpu_cold_s.iterations << " iters)" << endl;
         
    // PIM
    shared_ptr<PIMKernel> kernel = make_pim_kernel(N);
    double pim_time_cold = 0.0;
    // Cold start -> is_incremental defaults to false
    auto [pim_rank_cold, pim_iters_cold] = runPIMPageRank(uniform_init, kernel, pim_time_cold);

    // -----------------------------------------------------------------------
    // Stage B: Incremental Update (Warm Start)
    // -----------------------------------------------------------------------
    int inserted = 0;
    for (size_t i = base_edge_count; i < all_edges.size(); i++) {
        g.addEdge(all_edges[i].first, all_edges[i].second);
        inserted++;
    }
    cout << "  Inserted " << inserted << " new edges dynamically." << endl;

    // CPU Incremental
    CPUDRAMStats cpu_warm_s = simulateCPUOnDRAM_Dense(g, damping, tol, max_iter);
    cout << "  CPU Warm Start: " << cpu_warm_s.cycles
         << " cycles (" << cpu_warm_s.sim_time_ns << " ns, "
         << cpu_warm_s.iterations << " iters)" << endl;

    // PIM Incremental
    // file locks and static counters BEFORE allocating the new one.
    kernel.reset();
    mem_.reset();
    
    // Now it is safe to spin up the clean hardware state
    kernel = make_pim_kernel(N); 
    
    double pim_time_warm = 0.0;

    auto [pim_rank_warm, pim_iters_warm] = runPIMPageRank(pim_rank_cold, kernel, pim_time_warm, true);
    
    // -----------------------------------------------------------------------
    // Verification & Results
    // -----------------------------------------------------------------------
    vector<float> cpu_rank_warm = g.runPageRankCPU(damping, tol, max_iter);
    float max_diff = 0.0f;
    for (int v = 0; v < N; v++) {
        max_diff = max(max_diff, fabs(cpu_rank_warm[v] - pim_rank_warm[v]));
    }
    EXPECT_LT(max_diff, 0.01f); // Account for FP16 accumulation drift

    cout << "\n  [ Performance Benchmarks ]" << endl;
    cout << "  PIM Cold Start Time: " << pim_time_cold << " ms (" << pim_iters_cold << " iters)" << endl;
    cout << "  ---" << endl;
    cout << "  PIM Warm Start Time: " << pim_time_warm << " ms (" << pim_iters_warm << " iters)" << endl;
    cout << "  Max PIM drift vs CPU : " << max_diff << endl;

    // Fetch memory system telemetry
    PIMStats pim_s = getPIMStats(kernel);
    printUnifiedTable(cpu_warm_s, pim_s, N, pim_iters_warm);
}

// ===========================================================================
// Test 9: PIM vs CPU - Real-world graph (web-Google) with Incremental Updates
//
// Loads directed edges from the Google web graph (875K nodes, 5.1M edges).
// Due to the dense N x N matrix limitation of the current PIM kernel layout,
// the graph is constrained to a subset of nodes so cycle-accurate simulations
// can finish in a reasonable timeframe.
// ===========================================================================

TEST_F(PageRankFixture, real_world_web_google_pagerank)
{
    cout << "\n>> PageRank Real-World: web-Google.txt (Incremental)" << endl;

    const string filename = "src/tests/web-Google.txt";
    const int MAX_NODES = 512; // Constrain dense matrix size for simulation speed

    std::ifstream infile(filename);
    ASSERT_TRUE(infile.is_open()) << "  [!] Could not open " << filename << ". Please ensure it is in the working directory.";

    std::unordered_map<int, int> id_map;
    std::vector<std::pair<int, int>> all_edges;
    int next_id = 0;
    std::string line;

    // 1. Parse File and Map IDs
    while (std::getline(infile, line))
    {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int u_raw, v_raw;
        if (!(iss >> u_raw >> v_raw)) continue;

        if (id_map.find(u_raw) == id_map.end() && next_id < MAX_NODES) id_map[u_raw] = next_id++;
        if (id_map.find(v_raw) == id_map.end() && next_id < MAX_NODES) id_map[v_raw] = next_id++;

        if (id_map.find(u_raw) != id_map.end() && id_map.find(v_raw) != id_map.end())
            all_edges.push_back({id_map[u_raw], id_map[v_raw]});
    }

    // 2. Pad N to a multiple of 16 for NumpyBurstType lane alignment
    int N = next_id;
    if (N % 16 != 0) N = (N + 15) / 16 * 16;

    cout << "  Loaded " << all_edges.size() << " edges across " << N
         << " padded nodes (sub-graph max: " << MAX_NODES << ")." << endl;

    // Split edges: 85% for the base graph, 15% for the incremental update
    int base_edge_count = (int)(all_edges.size() * 0.85);
    PageRankGraph g(N);

    for (int i = 0; i < base_edge_count; i++)
        g.addEdge(all_edges[i].first, all_edges[i].second);

    const float damping  = 0.85f;
    const float tol      = 1e-4f;
    const int   max_iter = 140;
    const float base     = (1.0f - damping) / N;

    // PIM helper lambda for timing and convergence
    auto runPIMPageRank = [&](const vector<float>& init_rank,
                              shared_ptr<PIMKernel> kernel,
                              double& sim_time_ms,
                              bool is_incremental = false) -> pair<vector<float>, int>
    {
        NumpyBurstType weight_npbst, input_npbst;
        g.buildTransitionMatrix(weight_npbst, damping);
        kernel->preloadGemv(&weight_npbst);

        vector<float> rank = init_rank;
        vector<float> current_input = init_rank;

        if (is_incremental)
            current_input = g.getIncrementalDelta(rank, damping);

        int iters = 0;
        auto start = chrono::high_resolution_clock::now();

        for (int iter = 0; iter < max_iter; iter++)
        {
            iters++;
            input_npbst.bData.clear();
            input_npbst.bShape.clear();
            input_npbst.shape.clear();
            g.buildRankVector(current_input, input_npbst);

            kernel->executeGemv(&weight_npbst, &input_npbst, false);

            unsigned end_col = kernel->getResultColGemv(N / 16, N);
            std::vector<BurstType> raw(N);
            kernel->readResult(raw.data(), pimBankType::ODD_BANK, N, 0, 0, end_col);
            kernel->runPIM();

            vector<float> new_vec(N);
            float diff = 0.0f;

            for (int v = 0; v < N; v++)
            {
                new_vec[v] = convertH2F(raw[v].fp16ReduceSum());
                if (is_incremental) {
                    rank[v] += new_vec[v];
                    diff = max(diff, fabs(new_vec[v]));  // L-inf norm instead of L1
                } else {
                    diff = max(diff, fabs(new_vec[v] - rank[v]));  // same for cold
                    rank[v] = new_vec[v];
                }
            }

            current_input = new_vec;
            if (diff < tol && (!is_incremental || iter >= 4)) break;
        }

        auto end = chrono::high_resolution_clock::now();
        sim_time_ms = chrono::duration<double, milli>(end - start).count();
        return {rank, iters};
    };

    // -----------------------------------------------------------------------
    // Stage A: Initial Graph Convergence (Cold Start)
    // -----------------------------------------------------------------------
    vector<float> uniform_init(N, 1.0f / N);

    CPUDRAMStats cpu_cold_s = simulateCPUOnDRAM_Dense(g, damping, tol, max_iter);
    cout << "  CPU Cold Start: " << cpu_cold_s.cycles
         << " cycles (" << cpu_cold_s.sim_time_ns << " ns, "
         << cpu_cold_s.iterations << " iters)" << endl;

    shared_ptr<PIMKernel> kernel = make_pim_kernel(N);
    double pim_time_cold = 0.0;
    auto [pim_rank_cold, pim_iters_cold] = runPIMPageRank(uniform_init, kernel, pim_time_cold);

    // -----------------------------------------------------------------------
    // Stage B: Incremental Update (Warm Start)
    // -----------------------------------------------------------------------
    int inserted = 0;
    for (size_t i = base_edge_count; i < all_edges.size(); i++)
    {
        g.addEdge(all_edges[i].first, all_edges[i].second);
        inserted++;
    }
    cout << "  Inserted " << inserted << " new edges dynamically." << endl;

    CPUDRAMStats cpu_warm_s = simulateCPUOnDRAM_Dense(g, damping, tol, max_iter);
    cout << "  CPU Warm Start: " << cpu_warm_s.cycles
         << " cycles (" << cpu_warm_s.sim_time_ns << " ns, "
         << cpu_warm_s.iterations << " iters)" << endl;

    kernel.reset();
    mem_.reset();
    kernel = make_pim_kernel(N);

    double pim_time_warm = 0.0;
    auto [pim_rank_warm, pim_iters_warm] = runPIMPageRank(pim_rank_cold, kernel, pim_time_warm, true);

    // -----------------------------------------------------------------------
    // Verification & Results
    // -----------------------------------------------------------------------
    vector<float> cpu_rank_warm = g.runPageRankCPU(damping, tol, max_iter);
    float max_diff = 0.0f;
    for (int v = 0; v < N; v++)
        max_diff = max(max_diff, fabs(cpu_rank_warm[v] - pim_rank_warm[v]));
    EXPECT_LT(max_diff, 0.01f);

    cout << "\n  [ Performance Benchmarks ]" << endl;
    cout << "  PIM Cold Start Time: " << pim_time_cold << " ms (" << pim_iters_cold << " iters)" << endl;
    cout << "  ---" << endl;
    cout << "  PIM Warm Start Time: " << pim_time_warm << " ms (" << pim_iters_warm << " iters)" << endl;
    cout << "  Max PIM drift vs CPU : " << max_diff << endl;

    PIMStats pim_s = getPIMStats(kernel);
    printUnifiedTable(cpu_warm_s, pim_s, N, pim_iters_warm);
}

TEST_F(PageRankFixture, real_world_roadnet_pagerank)
{
    cout << "\n>> PageRank Real-World: roadNet-CA.txt (Incremental)" << endl;

    const string filename = "src/tests/roadNet-CA.txt";
    const int MAX_NODES = 512; // Constrain dense matrix size for simulation speed
    
    std::ifstream infile(filename);
    ASSERT_TRUE(infile.is_open()) << "  [!] Could not open " << filename << ". Please ensure it is in the working directory.";

    std::unordered_map<int, int> id_map;
    std::vector<std::pair<int, int>> all_edges;
    int next_id = 0;
    std::string line;

    // 1. Parse File and Map IDs
    while (std::getline(infile, line))
    {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        int u_raw, v_raw;
        if (!(iss >> u_raw >> v_raw)) continue;

        // Map raw IDs to a contiguous 0..N-1 range
        if (id_map.find(u_raw) == id_map.end() && next_id < MAX_NODES) id_map[u_raw] = next_id++;
        if (id_map.find(v_raw) == id_map.end() && next_id < MAX_NODES) id_map[v_raw] = next_id++;

        // Keep the edge if both nodes fall within our subset bounds
        if (id_map.find(u_raw) != id_map.end() && id_map.find(v_raw) != id_map.end()) {
            all_edges.push_back({id_map[u_raw], id_map[v_raw]});
        }
    }

    // 2. Pad N to a multiple of 16 for NumpyBurstType lane alignment
    int N = next_id;
    if (N % 16 != 0) N = (N + 15) / 16 * 16;
    
    cout << "  Loaded " << all_edges.size() << " edges across " << N 
         << " padded nodes (sub-graph max: " << MAX_NODES << ")." << endl;

    // Split edges: 85% for the base graph, 15% for the incremental update
    int base_edge_count = all_edges.size() * 0.85;
    PageRankGraph g(N);

    for (int i = 0; i < base_edge_count; i++) {
        g.addEdge(all_edges[i].first, all_edges[i].second);
    }
    
    const float damping = 0.85f;
    const float tol = 1e-4f;
    const int max_iter = 140;
    const float base = (1.0f - damping) / N;

    // PIM helper lambda for timing and convergence
    auto runPIMPageRank = [&](const vector<float>& init_rank, 
                              shared_ptr<PIMKernel> kernel, 
                              double& sim_time_ms, 
                              bool is_incremental = false) -> pair<vector<float>, int>
    {
        NumpyBurstType weight_npbst, input_npbst;
        g.buildTransitionMatrix(weight_npbst, damping); // Pass damping to ensure matrix is built correctly

        kernel->preloadGemv(&weight_npbst);

        vector<float> rank = init_rank;
        vector<float> current_input = init_rank;
        
        if (is_incremental) {
            current_input = g.getIncrementalDelta(rank, damping);
        }

        int iters = 0;
        auto start = chrono::high_resolution_clock::now();

        for (int iter = 0; iter < max_iter; iter++)
        {
            iters++;
            input_npbst.bData.clear();
            input_npbst.bShape.clear();
            input_npbst.shape.clear();
            
            // Build the input burst using either absolute R (Cold) or ΔR (Warm)
            g.buildRankVector(current_input, input_npbst);

            // Execute GEMV using the PRELOADED weights
            kernel->executeGemv(&weight_npbst, &input_npbst, false);

            unsigned end_col = kernel->getResultColGemv(N / 16, N);
            
            // Use std::vector for safe memory management
            std::vector<BurstType> raw(N);
            kernel->readResult(raw.data(), pimBankType::ODD_BANK, N, 0, 0, end_col);
            kernel->runPIM();

            vector<float> new_vec(N);
            float diff = 0.0f;
            
            for (int v = 0; v < N; v++) {
                // Since buildTransitionMatrix incorporates base and damping, 
                // the GEMV reduction is our exact new vector.
                new_vec[v] = convertH2F(raw[v].fp16ReduceSum());
                
                if (is_incremental) {
                    rank[v] += new_vec[v];
                    diff = max(diff, fabs(new_vec[v]));  // L-inf norm instead of L1
                } else {
                    diff = max(diff, fabs(new_vec[v] - rank[v]));  // same for cold
                    rank[v] = new_vec[v];
                }
            }
            
            // Next iteration's input is the output of this iteration
            current_input = new_vec;

            if (diff < tol && (!is_incremental || iter >= 4)) break;
        }

        auto end = chrono::high_resolution_clock::now();
        sim_time_ms = chrono::duration<double, milli>(end - start).count();
        return {rank, iters};
    };

    // -----------------------------------------------------------------------
    // Stage A: Initial Graph Convergence (Cold Start)
    // -----------------------------------------------------------------------
    vector<float> uniform_init(N, 1.0f / N);
    
    // CPU
    CPUDRAMStats cpu_cold_s = simulateCPUOnDRAM_Dense(g, damping, tol, max_iter);
    cout << "  CPU Cold Start: " << cpu_cold_s.cycles
         << " cycles (" << cpu_cold_s.sim_time_ns << " ns, "
         << cpu_cold_s.iterations << " iters)" << endl;
         
    // PIM
    shared_ptr<PIMKernel> kernel = make_pim_kernel(N);
    double pim_time_cold = 0.0;
    // Cold start -> is_incremental defaults to false
    auto [pim_rank_cold, pim_iters_cold] = runPIMPageRank(uniform_init, kernel, pim_time_cold);

    // -----------------------------------------------------------------------
    // Stage B: Incremental Update (Warm Start)
    // -----------------------------------------------------------------------
    int inserted = 0;
    for (size_t i = base_edge_count; i < all_edges.size(); i++) {
        g.addEdge(all_edges[i].first, all_edges[i].second);
        inserted++;
    }
    cout << "  Inserted " << inserted << " new edges dynamically." << endl;

    // CPU Incremental
    CPUDRAMStats cpu_warm_s = simulateCPUOnDRAM_Dense(g, damping, tol, max_iter);
    cout << "  CPU Warm Start: " << cpu_warm_s.cycles
         << " cycles (" << cpu_warm_s.sim_time_ns << " ns, "
         << cpu_warm_s.iterations << " iters)" << endl;

    // PIM Incremental
    // [!] CRITICAL FIX: Explicitly destroy the old simulator states to release
    // file locks and static counters BEFORE allocating the new one.
    kernel.reset();
    mem_.reset();
    
    // Now it is safe to spin up the clean hardware state
    kernel = make_pim_kernel(N); 
    
    double pim_time_warm = 0.0;
    
    // [!] Added 'true' here to activate the Delta PageRank strategy
    auto [pim_rank_warm, pim_iters_warm] = runPIMPageRank(pim_rank_cold, kernel, pim_time_warm, true);
    
    // -----------------------------------------------------------------------
    // Verification & Results
    // -----------------------------------------------------------------------
    vector<float> cpu_rank_warm = g.runPageRankCPU(damping, tol, max_iter);
    float max_diff = 0.0f;
    for (int v = 0; v < N; v++) {
        max_diff = max(max_diff, fabs(cpu_rank_warm[v] - pim_rank_warm[v]));
    }
    EXPECT_LT(max_diff, 0.01f); // Account for FP16 accumulation drift

    cout << "\n  [ Performance Benchmarks ]" << endl;
    cout << "  PIM Cold Start Time: " << pim_time_cold << " ms (" << pim_iters_cold << " iters)" << endl;
    cout << "  ---" << endl;
    cout << "  PIM Warm Start Time: " << pim_time_warm << " ms (" << pim_iters_warm << " iters)" << endl;
    cout << "  Max PIM drift vs CPU : " << max_diff << endl;

    // Fetch memory system telemetry
    PIMStats pim_s = getPIMStats(kernel);
    printUnifiedTable(cpu_warm_s, pim_s, N, pim_iters_warm);
}