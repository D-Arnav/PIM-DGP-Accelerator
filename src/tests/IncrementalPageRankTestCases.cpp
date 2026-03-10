/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#include "tests/IncrementalPageRank.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "Burst.h"
#include "MultiChannelMemorySystem.h"
#include "gtest/gtest.h"

namespace
{
struct IncrementalPageRankDataset
{
    static constexpr size_t kNumNodes = 4096;
    static constexpr size_t kBaseEdgeCount = 1024 * 1024;
    static constexpr size_t kUpdateEdgeCount = 64 * 1024;

    DRAMSim::NumpyBurstType base_src;
    DRAMSim::NumpyBurstType base_dst;
    DRAMSim::NumpyBurstType update_src;
    DRAMSim::NumpyBurstType update_dst;
    DRAMSim::NumpyBurstType expected_ranks;

    void load()
    {
        base_src.loadFp32("data/ipr/ipr_base_src_1048576.npy");
        base_dst.loadFp32("data/ipr/ipr_base_dst_1048576.npy");
        update_src.loadFp32("data/ipr/ipr_update_src_65536.npy");
        update_dst.loadFp32("data/ipr/ipr_update_dst_65536.npy");
        expected_ranks.loadFp32("data/ipr/ipr_expected_4096_1048576_65536.npy");
    }
};

class IncrementalPageRankTrafficModel
{
  public:
    IncrementalPageRankTrafficModel()
        : mem_(std::make_shared<DRAMSim::MultiChannelMemorySystem>(
              "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".", "example_app",
              256 * 64 * 2)),
          next_addr_(0),
          burst_bytes_(getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") /
                       8)
    {
    }

    uint64_t simulateSolve(size_t iterations, size_t total_edge_count, size_t update_edge_count,
                           size_t num_nodes, bool is_pim)
    {
        const size_t rank_bytes = num_nodes * sizeof(float);
        const size_t rank_bursts = bytesToBursts(rank_bytes);
        uint64_t total_cycles = 0;

        for (size_t iteration = 0; iteration < iterations; ++iteration)
        {
            const size_t active_edges = (is_pim && iteration == 0) ? update_edge_count : total_edge_count;
            const size_t edge_bursts = bytesToBursts(active_edges * sizeof(uint32_t) * 2);

            issueBursts(edge_bursts, false);
            issueBursts(rank_bursts, false);
            total_cycles += drain();

            issueBursts(rank_bursts, true);
            total_cycles += drain();
        }

        return total_cycles;
    }

  private:
    size_t bytesToBursts(size_t num_bytes) const
    {
        return (num_bytes + burst_bytes_ - 1) / burst_bytes_;
    }

    void issueBursts(size_t num_bursts, bool is_write)
    {
        DRAMSim::BurstType null_burst;
        for (size_t burst = 0; burst < num_bursts; ++burst)
        {
            mem_->addTransaction(is_write, next_addr_, &null_burst);
            next_addr_ += burst_bytes_;
        }
    }

    uint64_t drain()
    {
        uint64_t cycles = 0;
        while (mem_->hasPendingTransactions())
        {
            mem_->update();
            cycles++;
        }
        return cycles;
    }

    std::shared_ptr<DRAMSim::MultiChannelMemorySystem> mem_;
    uint64_t next_addr_;
    size_t burst_bytes_;
};

void addEdges(IncrementalPageRankModel& model, const DRAMSim::NumpyBurstType& src,
              const DRAMSim::NumpyBurstType& dst)
{
    ASSERT_EQ(src.data.size(), dst.data.size());
    for (size_t idx = 0; idx < src.data.size(); ++idx)
        model.addEdge(static_cast<size_t>(src.data[idx]), static_cast<size_t>(dst.data[idx]));
}

double sumRanks(const std::vector<double>& ranks)
{
    return std::accumulate(ranks.begin(), ranks.end(), 0.0);
}

void expectNearExpected(const PageRankResult& result, const DRAMSim::NumpyBurstType& expected)
{
    ASSERT_EQ(result.ranks.size(), expected.data.size());
    for (size_t idx = 0; idx < result.ranks.size(); ++idx)
        EXPECT_NEAR(result.ranks[idx], expected.data[idx], 1e-5);
}

class IncrementalPageRankBenchTestCase
{
  public:
    IncrementalPageRankBenchTestCase()
    {
        dataset_.load();
    }

    uint64_t measureCycle(bool is_pim)
    {
        IncrementalPageRankModel model(IncrementalPageRankDataset::kNumNodes);
        addEdges(model, dataset_.base_src, dataset_.base_dst);

        const PageRankResult initial = model.runFromScratch();
        EXPECT_EQ(model.graph().edgeCount(), IncrementalPageRankDataset::kBaseEdgeCount);

        addEdges(model, dataset_.update_src, dataset_.update_dst);

        const PageRankResult updated = is_pim ? model.runIncremental() : model.runFromScratch();

        EXPECT_NEAR(sumRanks(updated.ranks), 1.0, 1e-9);
        expectNearExpected(updated, dataset_.expected_ranks);

        IncrementalPageRankTrafficModel traffic_model;
        return traffic_model.simulateSolve(updated.iterations,
                                           IncrementalPageRankDataset::kBaseEdgeCount +
                                               IncrementalPageRankDataset::kUpdateEdgeCount,
                                           IncrementalPageRankDataset::kUpdateEdgeCount,
                                           IncrementalPageRankDataset::kNumNodes, is_pim);
    }

    void printTestMessage(bool is_pim) const
    {
        cout << "  IPR (PIM " << (is_pim ? "enabled)" : "disabled)") << endl;
        cout << "  Nodes : " << IncrementalPageRankDataset::kNumNodes << endl;
        cout << "  Base edges : " << IncrementalPageRankDataset::kBaseEdgeCount << endl;
        cout << "  Update edges : " << IncrementalPageRankDataset::kUpdateEdgeCount << endl;
    }

  private:
    IncrementalPageRankDataset dataset_;
};

class IncrementalPageRankBenchFixture : public testing::Test
{
  public:
    void SetUp() override
    {
        pim_cycle_ = 0;
        non_pim_cycle_ = 0;
        printTestMessage();
    }

    void TearDown() override
    {
        printResult(non_pim_cycle_ / static_cast<float>(pim_cycle_));
    }

    void executeKernel()
    {
        bench_.printTestMessage(false);
        non_pim_cycle_ = bench_.measureCycle(false);
        printStats(non_pim_cycle_);
    }

    void executePIMKernel()
    {
        bench_.printTestMessage(true);
        pim_cycle_ = bench_.measureCycle(true);
        printStats(pim_cycle_);
    }

    void expectPIMBench(float expected_perf_gain)
    {
        EXPECT_TRUE(static_cast<float>(non_pim_cycle_) / pim_cycle_ > expected_perf_gain)
            << static_cast<float>(non_pim_cycle_) / pim_cycle_ << endl;
    }

  private:
    void printTestMessage() const
    {
        cout << ">>Performance Test" << endl;
    }

    void printStats(uint64_t cycle) const
    {
        cout << "> Test Results " << endl;
        cout << "> Cycle : " << cycle << endl;
        cout << endl;
    }

    void printResult(float gain) const
    {
        cout << "> Speed-up : " << gain << endl;
    }

    IncrementalPageRankBenchTestCase bench_;
    uint64_t pim_cycle_;
    uint64_t non_pim_cycle_;
};
}  // namespace

TEST(IncrementalPageRank, IPR)
{
    IncrementalPageRankDataset dataset;
    dataset.load();

    IncrementalPageRankModel model(IncrementalPageRankDataset::kNumNodes);
    addEdges(model, dataset.base_src, dataset.base_dst);

    const PageRankResult initial = model.runFromScratch();
    ASSERT_EQ(initial.ranks.size(), IncrementalPageRankDataset::kNumNodes);
    EXPECT_NEAR(sumRanks(initial.ranks), 1.0, 1e-9);
    EXPECT_GT(initial.edge_visits, 0u);
    EXPECT_EQ(model.graph().edgeCount(), IncrementalPageRankDataset::kBaseEdgeCount);

    addEdges(model, dataset.update_src, dataset.update_dst);

    const PageRankResult updated = model.runFromScratch();
    EXPECT_NEAR(sumRanks(updated.ranks), 1.0, 1e-9);
    EXPECT_GT(updated.iterations, 0u);
    EXPECT_EQ(model.graph().edgeCount(),
              IncrementalPageRankDataset::kBaseEdgeCount +
                  IncrementalPageRankDataset::kUpdateEdgeCount);
    expectNearExpected(updated, dataset.expected_ranks);

    IncrementalPageRankTrafficModel traffic_model;
    const uint64_t baseline_cycles =
        traffic_model.simulateSolve(updated.iterations, model.graph().edgeCount(),
                                    IncrementalPageRankDataset::kUpdateEdgeCount,
                                    IncrementalPageRankDataset::kNumNodes, false);
    EXPECT_GT(baseline_cycles, 0u);
}

TEST(IncrementalPageRank, IPR_PIM)
{
    IncrementalPageRankDataset dataset;
    dataset.load();

    IncrementalPageRankModel baseline(IncrementalPageRankDataset::kNumNodes);
    IncrementalPageRankModel pim_accelerated(IncrementalPageRankDataset::kNumNodes);
    addEdges(baseline, dataset.base_src, dataset.base_dst);
    addEdges(pim_accelerated, dataset.base_src, dataset.base_dst);

    const PageRankResult baseline_initial = baseline.runFromScratch();
    const PageRankResult pim_initial = pim_accelerated.runFromScratch();
    for (size_t idx = 0; idx < baseline_initial.ranks.size(); ++idx)
        EXPECT_NEAR(baseline_initial.ranks[idx], pim_initial.ranks[idx], 1e-10);

    addEdges(baseline, dataset.update_src, dataset.update_dst);
    addEdges(pim_accelerated, dataset.update_src, dataset.update_dst);

    const PageRankResult baseline_result = baseline.runFromScratch();
    const PageRankResult pim_result = pim_accelerated.runIncremental();

    IncrementalPageRankTrafficModel baseline_traffic;
    IncrementalPageRankTrafficModel pim_traffic;
    const uint64_t baseline_cycles =
        baseline_traffic.simulateSolve(baseline_result.iterations, baseline.graph().edgeCount(),
                                       IncrementalPageRankDataset::kUpdateEdgeCount,
                                       IncrementalPageRankDataset::kNumNodes, false);
    const uint64_t pim_cycles =
        pim_traffic.simulateSolve(pim_result.iterations, pim_accelerated.graph().edgeCount(),
                                  IncrementalPageRankDataset::kUpdateEdgeCount,
                                  IncrementalPageRankDataset::kNumNodes, true);

    EXPECT_NEAR(sumRanks(baseline_result.ranks), 1.0, 1e-9);
    EXPECT_NEAR(sumRanks(pim_result.ranks), 1.0, 1e-9);

    for (size_t idx = 0; idx < baseline_result.ranks.size(); ++idx)
        EXPECT_NEAR(baseline_result.ranks[idx], pim_result.ranks[idx], 1e-8);

    expectNearExpected(baseline_result, dataset.expected_ranks);
    expectNearExpected(pim_result, dataset.expected_ranks);
    EXPECT_LT(pim_result.edge_visits, baseline_result.edge_visits);
    EXPECT_LT(pim_cycles, baseline_cycles);
}

TEST_F(IncrementalPageRankBenchFixture, ipr)
{
    executeKernel();
    executePIMKernel();
    expectPIMBench(1.0f);
}