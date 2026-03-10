import numpy as np

NUM_NODES = 4096
BASE_DEGREE = 256
UPDATE_DEGREE = 16
DAMPING = 0.85
TOLERANCE = 1e-10
MAX_ITERATIONS = 200


def generate_edges():
    base_src = np.empty(NUM_NODES * BASE_DEGREE, dtype=np.float32)
    base_dst = np.empty(NUM_NODES * BASE_DEGREE, dtype=np.float32)
    update_src = np.empty(NUM_NODES * UPDATE_DEGREE, dtype=np.float32)
    update_dst = np.empty(NUM_NODES * UPDATE_DEGREE, dtype=np.float32)

    base_idx = 0
    update_idx = 0
    for src in range(NUM_NODES):
        base_neighbors = set()
        for k in range(BASE_DEGREE):
            dst = (src * 131 + k * 17 + 1) % NUM_NODES
            if dst == src:
                dst = (dst + 1) % NUM_NODES
            while dst in base_neighbors or dst == src:
                dst = (dst + 17) % NUM_NODES
            base_neighbors.add(dst)
            base_src[base_idx] = src
            base_dst[base_idx] = dst
            base_idx += 1

        update_neighbors = set()
        t = 0
        while len(update_neighbors) < UPDATE_DEGREE:
            dst = (src * 193 + t * 29 + 7) % NUM_NODES
            t += 1
            if dst == src or dst in base_neighbors or dst in update_neighbors:
                continue
            update_neighbors.add(dst)
            update_src[update_idx] = src
            update_dst[update_idx] = dst
            update_idx += 1

    return base_src, base_dst, update_src, update_dst


def build_adjacency(src, dst):
    adjacency = [[] for _ in range(NUM_NODES)]
    for src_node, dst_node in zip(src.astype(np.int32), dst.astype(np.int32)):
        adjacency[src_node].append(dst_node)
    return adjacency


def pagerank(adjacency):
    ranks = np.full(NUM_NODES, 1.0 / NUM_NODES, dtype=np.float64)
    base = (1.0 - DAMPING) / NUM_NODES

    for _ in range(MAX_ITERATIONS):
        next_ranks = np.full(NUM_NODES, base, dtype=np.float64)
        dangling_mass = 0.0

        for src, neighbors in enumerate(adjacency):
            if not neighbors:
                dangling_mass += ranks[src]
                continue

            contribution = DAMPING * ranks[src] / len(neighbors)
            for dst in neighbors:
                next_ranks[dst] += contribution

        next_ranks += DAMPING * dangling_mass / NUM_NODES
        if np.abs(next_ranks - ranks).sum() < TOLERANCE:
            ranks = next_ranks
            break
        ranks = next_ranks

    ranks /= ranks.sum()
    return ranks.astype(np.float32)


def main():
    base_src, base_dst, update_src, update_dst = generate_edges()

    np.save("ipr_base_src_1048576.npy", base_src)
    np.save("ipr_base_dst_1048576.npy", base_dst)
    np.save("ipr_update_src_65536.npy", update_src)
    np.save("ipr_update_dst_65536.npy", update_dst)

    adjacency = build_adjacency(base_src, base_dst)
    for src_node, dst_node in zip(update_src.astype(np.int32), update_dst.astype(np.int32)):
        adjacency[src_node].append(dst_node)

    expected = pagerank(adjacency)
    np.save("ipr_expected_4096_1048576_65536.npy", expected)


if __name__ == "__main__":
    main()