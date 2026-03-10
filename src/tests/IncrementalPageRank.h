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

#ifndef __INCREMENTAL_PAGE_RANK_H__
#define __INCREMENTAL_PAGE_RANK_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

struct PageRankResult
{
    std::vector<double> ranks;
    size_t iterations = 0;
    size_t edge_visits = 0;
};

class DynamicGraph
{
  public:
    explicit DynamicGraph(size_t num_nodes)
        : out_neighbors_(num_nodes), has_edge_(num_nodes, std::vector<bool>(num_nodes, false))
    {
    }

    size_t numNodes() const
    {
        return out_neighbors_.size();
    }

    bool addEdge(size_t src, size_t dst)
    {
        if (src >= numNodes() || dst >= numNodes())
            throw std::out_of_range("Edge endpoint out of range");
        if (has_edge_[src][dst])
            return false;
        has_edge_[src][dst] = true;
        out_neighbors_[src].push_back(dst);
        return true;
    }

    const std::vector<size_t>& outNeighbors(size_t node) const
    {
        return out_neighbors_[node];
    }

    size_t outDegree(size_t node) const
    {
        return out_neighbors_[node].size();
    }

    size_t edgeCount() const
    {
        size_t count = 0;
        for (const auto& neighbors : out_neighbors_)
            count += neighbors.size();
        return count;
    }

  private:
    std::vector<std::vector<size_t>> out_neighbors_;
    std::vector<std::vector<bool>> has_edge_;
};

class IncrementalPageRankModel
{
  public:
    explicit IncrementalPageRankModel(size_t num_nodes, double damping = 0.85,
                                      double tolerance = 1e-10, size_t max_iterations = 200)
        : graph_(num_nodes),
          damping_(damping),
          tolerance_(tolerance),
          max_iterations_(max_iterations),
          pending_update_edges_(0),
          previous_ranks_(num_nodes, 1.0 / static_cast<double>(num_nodes))
    {
    }

    bool addEdge(size_t src, size_t dst)
    {
        const bool added = graph_.addEdge(src, dst);
        if (added)
            pending_update_edges_++;
        return added;
    }

    const DynamicGraph& graph() const
    {
        return graph_;
    }

    PageRankResult runFromScratch()
    {
        std::vector<double> initial(graph_.numNodes(), 1.0 / static_cast<double>(graph_.numNodes()));
        return solve(initial, false);
    }

    PageRankResult runIncremental()
    {
        return solve(previous_ranks_, true);
    }

  private:
    PageRankResult solve(const std::vector<double>& initial_ranks, bool keep_state)
    {
        PageRankResult result;
        if (graph_.numNodes() == 0)
            return result;

        std::vector<double> ranks = normalize(initial_ranks);
        const double base_rank = (1.0 - damping_) / static_cast<double>(graph_.numNodes());

        for (size_t iteration = 0; iteration < max_iterations_; ++iteration)
        {
            std::vector<double> next(graph_.numNodes(), base_rank);
            double dangling_mass = 0.0;

            for (size_t src = 0; src < graph_.numNodes(); ++src)
            {
                const size_t out_degree = graph_.outDegree(src);
                if (out_degree == 0)
                {
                    dangling_mass += ranks[src];
                    continue;
                }

                const double contribution = damping_ * ranks[src] / static_cast<double>(out_degree);
                if (!(keep_state && iteration == 0))
                    result.edge_visits += out_degree;
                for (size_t dst : graph_.outNeighbors(src))
                    next[dst] += contribution;
            }

            if (keep_state && iteration == 0)
                result.edge_visits += pending_update_edges_;

            const double dangling_share = damping_ * dangling_mass /
                                          static_cast<double>(graph_.numNodes());
            for (double& value : next)
                value += dangling_share;

            double delta = 0.0;
            for (size_t idx = 0; idx < ranks.size(); ++idx)
                delta += std::abs(next[idx] - ranks[idx]);

            result.iterations = iteration + 1;
            ranks.swap(next);
            if (delta < tolerance_)
                break;
        }

        result.ranks = normalize(ranks);
        if (keep_state)
            previous_ranks_ = result.ranks;
        pending_update_edges_ = 0;
        return result;
    }

    std::vector<double> normalize(const std::vector<double>& values) const
    {
        std::vector<double> normalized = values;
        const double sum = std::accumulate(normalized.begin(), normalized.end(), 0.0);
        if (sum <= 0.0)
            return std::vector<double>(graph_.numNodes(), 1.0 / static_cast<double>(graph_.numNodes()));

        for (double& value : normalized)
            value /= sum;
        return normalized;
    }

    DynamicGraph graph_;
    double damping_;
    double tolerance_;
    size_t max_iterations_;
    size_t pending_update_edges_;
    std::vector<double> previous_ranks_;
};

#endif