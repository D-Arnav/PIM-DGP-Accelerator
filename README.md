# PIMSimulator

## Contents

  [1. Overview](#1-overview)  
  [2. HW Description](#2-hw-description)  
  [3. Setup](#3-setup)  
  [4. Programming Guide](#4-programming-guide)

## 1. Overview

PIMSimulator is a cycle accurate model that Single Instruction, Multiple Data (SIMD) execution units
that uses the bank-level parallelism in PIM Block to boost performance that would have otherwise used
multiple times of bandwidth from simultaneous access of all bank.
The simulator include memory and have embedded within it a PIM block, which consist of programmable
command registers, general purpose register files and execution units.

Based on https://github.com/umd-memsys/DRAMSim2, the simulator includes

* PIM Block:
  * Register files including CRF (for command), GRF (for vector value), SRF (for scalar value)
  * ALU (ADD, MUL, MAC, MAD, MOVE, FILL, NOP, JUMP, EXIT)
* PIM Kernel:
  * Generate a set of memory transactions for enabling PIM operation
* HBM2 support (refer to `ini/HBM2_samsung_2M_16B_x64.ini`)

## 2. HW description

PIM is a HBM stack that is pin compatible with HBM2 and have embedded within it a PIM block

### 2.1 Base Architecture

```C
|--------------|          |--------------|          |--------------|
|              |    (A)   |              |   (B)    |              |
|    HOST      |----------| Controller   |----------|    Memory    |
|              |          |              |          |              |
|--------------|          |--------------|          |--------------|
```

* Each channel is logically independent memory, so it has a dedicated independent controller.
* (A): Read [Addr], Write [Addr]
* (B): Activate, Read, Write, Precharge, Refresh, Activate_pim, ALU_pim, Precharge_pim, READ_pim

#### HBM2

* System Specification: system_hbm.ini
* HBM Specification: ini/HBM2_samsung_2M_16B_x64.ini
  * 1 PIM block per 2 banks, 4 Bank per Bankgroup, 4 Bank group per pseudo channel, 4 pseudo channel per die, 4 die per stack.
  * Prefetch size : 256bit
  * burst length: 4n
  * Pin speed: 2Gbps
  * The simulator supports the pseudo-channel mode only, and we assume that each pseudo-channel is totally independent.

### 2.2 Address mapping

* The address mapping is used when the memory controller decodes the address from host.
* Use Scheme8 addressing mode for PIM functionality.

```C
|<-rank->|<-row->|<-col high->|<-bg->|<-bank->|<-chan->|<-col low->|<-offset ->|
```
* the length of col_low is log(BL * JEDEC_DATA_BUS_BUTS/8), which are 5b both for HBM2
* You can also change the current addressing mode dynamically (Not recommended, though)

```C
// Static Setting in system_*.ini
ADDRESS_MAPPING_SCHEME=Scheme8
```
### 2.3 PIM Block Placement

* BANKS_PER_PIM_BLOCK = NUM_BANKS / NUM_PIM_BLOCKS

#### HBM2 case
```C
|--------|  |--------|
|        |  |        |
| BANK_0 |  | BANK_2 |
|        |  |        |
|--------|  |--------|
|  PB_0  |  |  PB_1  |
|--------|  |--------|
|        |  |        |
| BANK_1 |  | BANK_3 |
|        |  |        |
|--------|  |--------|
```
* A PIM Block (PB) is located per banks.
  * NUM_BANKS = 16, NUM_PIM_BLOCKS = 8


### 2.3 PIM Instruction-Set Architecture
|Type|Command|Description|Result (DST)|Operand (SRC0)|Operand (SRC1)|
|---|---|---|---|---|---|
|Arithmetic|ADD|addition |GRF|GRF, BANK, SRF|GRF, BANK, SRF|
|Arithmetic|MUL|multiplication |GRF|GRF, BANK|GRF, BANK, SRF|
|Arithmetic|MAC|multiply-accumulate |GRF_B|GRF, BANK|GRF, BANK, SRF|
|Arithmetic|MAD|multiply-and-add |GRF|GRF, BANK|GRF, BANK, SRF|
|Data|MOV|load or store data from register to bank|GRF, SRF|GRF, BANK||
|Data|FILL|copy data from bank to register|GRF, BANK|GRF, BANK||
|Control|NOP|do nothing||||
|Control|JUMP|jump instruction||||
|Control|EXIT|exit instruction||||

* Supports RISC-style 32-bit instructions
* Three instructions types
  * 4 Arithmetic: ADD, MUL, MAC, MAD
  * 2 Data transfer: MOV, FILL
  * 3 Control flows: NOP, JUMP, EXIT
* JUMP instruction
  * Zero-cycle static branch: supports only a pre-programmed numbers of iterations
* Operand type:
  * Vector Register (GRF_A, GRF_B)
  * Scalar Register (SRF)
  * Bank Row Buffer
* PIM instructions are stored in the Command Register File (CRF), and memory command triggers a CRF to perform a target instruction
  * each memory command increments the CRF PC
* DRAM commands decide where to retrieve data from DRAM for PIM arithmetic operations

### 2.4 Movement of Data
|Mode|Transaction|PIM Instruction|Operation|
|---|---|---|---|
|SB|Read|-|Normal Memory Read|
|SB|Write|-|Normal Memory Write|
|HAB|Write|-| PIM Write (Host to PIM Register)|
|PIM|-|MOV|read or write from bank to PIM Register|
|PIM|-|FILL|write from bank to PIM Registers|

* SB mode: standard DRAM operation
* HAB mode: Allowing concurrent accesses to multiple banks with a single DRAM command
* PIM mode: Triggers the execution of PIM instructions on the CRF by DRAM Command


## 3. Setup

### 3.1 Prerequisites
* `Scons` tool for compiling PIMSimulator:
```bash
sudo apt install scons
```
* `gtest` for running test cases:
```bash
sudo apt install libgtest-dev
```

### 3.2 Installing
* To Install PIMSimulator:
```bash
# compile
scons
```

### 3.3 Launch a Test Run
* Show a list of test cases
```bash
./sim --gtest_list_tests

# Example
PIMKernelFixture.
  gemv_tree
  gemv
  mul
  add
  relu
MemBandwidthFixture.
  hbm_read_bandwidth
  hbm_write_bandwidth
PIMBenchFixture.
  gemv
  mul
  add
  relu
```

* Test Running
```bash
# Running: functionality test (GEMV)
./sim --gtest_filter=PIMKernelFixture.gemv

# Running: functionality test (MUL)
./sim --gtest_filter=PIMKernelFixture.mul

# Running: performance test (GEMV)
./sim --gtest_filter=PIMBenchFixture.gemv

# Running: performance test (ADD)
./sim --gtest_filter=PIMBenchFixture.add
```

If you want to functionality test for other dimensions, generate a new dimension in `./data`
and add generated dimension to the source of `src/tests/KernelTestCases.cpp`.
Use the gen script in `./data` to generate data of the dimension to be changed.

### 3.4 Configuration

#### Turning on/off verbose mode
* You can select what kinds of log you want to see by modifying system_*.ini

#### Turning on/off data mode
* Data mode
  * build without -DNO_STORAGE option
* No-data mode
  * build with -DNO_STORAGE option
```bash
# build to No-data mode
scons NO_STORAGE=1
```

## 4 Programming Guide
Highly recommend you to refer to `src/tests/*` (especially, `src/tests/PIMKernel.cpp` and `src/tests/PIMBenchTestCases.cpp`)
To attach to host simulator, refer to `src/tests/PIMKernel.cpp`.
You can see commands that request memory transactions to the memory controller for GEMV or Eltwise operations on PIM.
It include a basic PIM procedure for GEMV operation in the `PIMKernel::executeGemv()`,
and also for Eltwise operation (add, mul, relu) in the `PIMKernel::executeEltwise()`

### 4.1 Primitive Function

```C
mem->addTransaction(is_read, address, tag, buffer);
```
* is_read: memory request types between READ('false') and WRITE('true')
* address: address used for memory / PIM transaction
* tag: Used for log or set to barrier. If not used, only three parameters are available,
        as `addTransaction(is_read, address, buffer)`
* buffer : used to verify pim functionality using data. Here, the buffer is at least 256-bit sized container.
If you do not want to use the data buffer, you can use it as below:
    ```C
    BurstType nullBst
    mem->addTransaction(isWrite, addr, &nullBst);
    ```

#### Memory transaction

* read
    ```C
    mem->addTransaction(false, addr, tag, buffer);
    ```
* write
    ```C
    mem->addTransaction(true, addr, tag, buffer);
    ```
Here, the buffer must be at least 256bit size container.

#### PIM transaction

* alu_pim (dataflow is similar to normal write)
    ```C
    mem->addTransaction(true, addr, tag, buffer);
    ```
  * Highly recommend you to refer to simple PIM operations using PIM ISA (`src/tests/PIMCmdGen.h`) and procedures using them(`src/tests/PIMKernel.cpp`)
  * The buffer must contain data to be broadcasted to all pim blocks of a specific channel.
  * In GEMV cacse, the corresponding weight is supplied from specific row, a specific col of multiple banks of a specific memory channel.
    * If each bank in a channel has unique ID, and bank addr in the transaction is BA, the banks satisfying (ID % BANKS_PER_PIM_BLOCK == BA) supply the weight to PIM blocks.
      * if BANKS_PER_PIM_BLOCK == 2, and BA = 0, bank 0,2,4,6,... supply the weight to pim-block 0,1,2,3,..., respectively.
      * if BANKS_PER_PIM_BLOCK == 2, and BA = 1, bank 1,3,5,7,... supply the weight to pim-block 0,1,2,3,..., respectively.
    * As a result, data broadcasted and data from multiple banks of a specific channel are multiplied and accumulated.

* read_pim (dataflow is similar to normal read)
    ```C
    mem->addTransaction(false, addr, tag, buffer);
    ```
   * Read the accumulated partial sum in the pim block. Then reset the buffer.
      * If each pim block in a channel has unique ID, and bank addr in the transaction is BA, the PIM block satisfying (ID == BA) supply partial sums to DQ.
   * Highly recommend to use the address that alu_pim command used at the last

### 4.2 PIM High-level Steps
* The following shows the high level steps of a generic PIM operation.
   * Place data in DRAM
   * Switch to HAB mode
   * Program CRF
   * Enable PIM
   * Execute PIM
   * Disable PIM
   * Switch to SB mode

* A similar procedure at the source level can be found in `src/tests/PIMKernel.cpp`.
```C
    /* Example Code - PIMKernel::executeELtwise() */

    parkIn();
    changePIMMode(dramMode::SB, dramMode::HAB);      // Switch to HAB
    programCrf(pim_cmds);                            // Program CRF
    changePIMMode(dramMode::HAB, dramMode::HAB_PIM); // Enable PIM

    if (ktype == KernelType::ADD || ktype == KernelType::MUL)
        computeAddOrMul(num_tile, input0_row, result_row, input1_row); // Execute PIM
    else if (ktype == KernelType::RELU)
        computeRelu(num_tile, input0_row, result_row);

    changePIMMode(dramMode::HAB_PIM, dramMode::HAB); // Disable PIM
    changePIMMode(dramMode::HAB, dramMode::SB);      // Switch to SB mode
    parkOut();

```

* The other basic operation flow on PIM for GEMV(Matrix Vector multiplication), Element-wise operation are described in the `src/tests/PIMKernel.cpp`.

### Contact
* Shin-haeng Kang (s-h.kang@samsung.com)
* Sanghoon Cha (s.h.cha@samsung.com)
* Seungwoo Seo (sgwoo.seo@samsung.com)
* Jin-seong kim (jseong82.kim@samsung.com)

---

## 5. PageRank on PIM 

This section covers the incremental PageRank implementation added for the EECS 573 final project.

### 5.1 What was implemented

Two versions of PageRank are implemented and compared:

| | CPU Baseline | PIM Accelerated |
|---|---|---|
| Precision | FP32 | FP16 |
| SpMV step | CPU (sparse, iterates edges only) | PIM GEMV kernel (dense N×N matrix) |
| Damping step | CPU | CPU |
| Incremental updates | Warm start from previous ranks | Warm start from previous ranks |

The key idea: PageRank repeatedly computes `rank_new = (1-d)/N + d * M * rank_old`. The expensive part is `M * rank_old` (a matrix-vector multiply). The PIM version offloads this to the GEMV kernel running inside HBM2 memory.

### 5.2 New files

| File | Description |
|---|---|
| `src/tests/PageRankGraph.h` | Graph data structure, CPU PageRank, transition matrix builder for PIM |
| `src/tests/PageRankTestCases.cpp` | 7 gtest tests covering correctness, incremental updates, and stats |

### 5.3 Build

Same as the rest of the simulator:

```bash
scons
```

### 5.4 Running the PageRank tests

```bash
# Run all PageRank tests
./sim --gtest_filter="PageRankFixture*"

# Run a specific test
./sim --gtest_filter="PageRankFixture.baseline_known_graph"
./sim --gtest_filter="PageRankFixture.stats_pim_vs_cpu"

# Run only real-world graph tests (require dataset files in src/tests/)
./sim --gtest_filter="PageRankFixture.real_world*"
```

> Note: The PIM tests simulate cycle-accurate HBM2 memory, so they are slow on a laptop (each synthetic test takes 5–15 seconds; real-world tests with N=512 can take 30–60 seconds).

### 5.5 Test descriptions

| Test | What it does |
|---|---|
| `baseline_known_graph` | Runs CPU PageRank on a small 4-node graph. Checks ranks sum to 1 and are all positive. |
| `baseline_incremental_edges` | Starts with a 64-node ring graph, inserts 10 random edges, verifies warm-start converges to the same result as cold-start. |
| `baseline_random_graph_timing` | Runs CPU PageRank on a random 256-node graph (avg_deg=8) and reports wall-clock time. |
| `pim_spmv_matches_cpu` | Runs one SpMV step on PIM and compares the result to the CPU. Passes if all 256 outputs are within 5% (FP16 tolerance). |
| `pim_full_pagerank` | Runs full PageRank to convergence using PIM for each SpMV step. Compares final ranks to CPU baseline. |
| `pim_incremental_edge_insertion` | Inserts 16 edges mid-run into a 256-node graph (avg_deg=6), then compares cold vs warm restart iteration counts on PIM. |
| `stats_pim_vs_cpu` | Runs both versions and prints a side-by-side stats table (cycles, memory transactions, data moved, bandwidth). CPU traffic is estimated analytically (dense and sparse models). |
| `stats_cpu_dram_simulated` | Routes the CPU's dense matrix memory access pattern through the same HBM2 DRAM simulator (no PIM ops), giving real simulated cycle counts for both sides so the comparison is apples-to-apples. |
| `real_world_incremental_pagerank` | Loads the cit-HepPh citation graph, runs cold-start PIM PageRank on the first 85% of edges, then inserts the remaining 15% and runs a warm (delta) restart. Compares final ranks to CPU baseline. |
| `real_world_web_google_pagerank` | Same two-stage cold/warm benchmark as above, using the web-Google web graph (875K nodes, 5.1M edges in the full dataset; capped at 512 nodes for simulation). |
| `real_world_roadnet_pagerank` | Same two-stage cold/warm benchmark using the roadNet-CA road network graph; capped at 512 nodes for simulation. |

### 5.6 Understanding the output

#### Correctness tests (tests 1–5)

```
[  PASSED  ] PageRankFixture.baseline_known_graph
```
A passing test means ranks are numerically correct within tolerance.

#### Incremental test output

```
Initial convergence:        12 iters
Cold restart (post-insert): 14 iters
Warm restart (incremental):  9 iters
```
- **Cold restart**: recomputes from uniform ranks after edge insertion
- **Warm restart**: recomputes starting from the previous ranks (incremental)
- Fewer iterations for warm restart = the benefit of incremental PageRank

#### Stats table output (test 7 — estimated CPU traffic)

```
╔══════════════════════════════════════════════════════╗
║          Stats Comparison  (N=256)                   ║
╠══════════════════════╦═══════════════════════════════╣
║ Metric               ║ CPU baseline  │ PIM           ║
╠══════════════════════╬═══════════════════════════════╣
║ Iterations           ║             8 │             8 ║
║ FLOPs (M)            ║          0.03 │             - ║
║ Simulated cycles     ║             - │         44154 ║
║ Simulated time (ns)  ║             - │       44154.0 ║
║ Memory reads (txns)  ║             - │        104960 ║
║ Memory writes (txns) ║             - │        676480 ║
║ Data moved (MB)      ║          0.08 │         23.85 ║
║ Memory BW (GB/s)     ║             - │        566.34 ║
╚══════════════════════╩═══════════════════════════════╝
```

The dashes on the CPU side exist because CPU runs on real hardware — the PIM simulator has no visibility into it. Test 8 fixes this.

#### Stats table output (test 8 — both sides through DRAM simulator)

Test 8 routes the CPU's sparse memory accesses through the same HBM2 DRAM simulator (without any PIM instructions), giving real simulated cycle counts on both sides.

```
╔══════════════════════════════════════════════════════════════╗
║     Stats Comparison (N=256, both through HBM2 DRAM sim)    ║
╠══════════════════════════╦═══════════════╦═════════════════╣
║ Metric                   ║ CPU (no PIM)  ║ PIM             ║
╠══════════════════════════╬═══════════════╬═════════════════╣
║ Iterations               ║             8 ║              10 ║
║ Simulated cycles         ║          1392 ║           44154 ║
║ Simulated time (ns)      ║        1392.0 ║         44154.0 ║
║ Memory reads (txns)      ║          2536 ║          104960 ║
║ Memory writes (txns)     ║           256 ║          676480 ║
║ Data moved (MB)          ║          0.09 ║           23.85 ║
║ Memory BW (GB/s)         ║         64.18 ║          566.34 ║
╠══════════════════════════╩═══════════════╩═════════════════╣
║ PIM cycle speedup: 0.03x                                    ║
╚════════════════════════════════════════════════════════════╝
```

| Field | What it means |
|---|---|
| **Simulated cycles** | Clock cycles in the simulated HBM2 hardware (not wall-clock time) |
| **Simulated time (ns)** | `cycles × tCK` — how long this would take on real HBM2 hardware |
| **Memory reads/writes (txns)** | Number of 32-byte burst transactions issued across all 64 channels |
| **Data moved (MB)** | Total bytes moved = transactions × 32 bytes |
| **Memory BW (GB/s)** | Effective bandwidth utilized (HBM2 peak is ~900 GB/s) |
| **PIM cycle speedup** | `CPU simulated time / PIM simulated time` |

**Key takeaway**: For a sparse graph (N=256, ~2000 edges), CPU is currently ~32x faster in simulated cycles because it only reads the edges it needs (~2500 transactions) while PIM loads the entire dense 256×256 matrix (~780,000 transactions). PIM does utilize 566 GB/s bandwidth (near HBM2 peak), showing the hardware works well — the bottleneck is the dense matrix representation. Switching to a sparse PIM kernel is the planned next optimization to close this gap.

### 5.7 Graph parameters

Synthetic tests generate graphs internally — no external files needed. Real-world tests require the dataset files to be present in `src/tests/`. Graph parameters are hardcoded per test:

| Test | N (vertices) | Avg out-degree | RNG seed | Source |
|---|---|---|---|---|
| `baseline_known_graph` | 4 | hand-crafted (5 edges) | — | synthetic |
| `baseline_incremental_edges` | 64 | 1 (ring) + 10 random edges | 99 | synthetic |
| `baseline_random_graph_timing` | 256 | 8 | 42 | synthetic |
| `pim_spmv_matches_cpu` | 256 | 8 | 7 | synthetic |
| `pim_full_pagerank` | 256 | 8 | 13 | synthetic |
| `pim_incremental_edge_insertion` | 256 | 6 (+ 16 inserted) | 17 / 55 | synthetic |
| `stats_pim_vs_cpu` | 256 | 8 | 42 | synthetic |
| `stats_cpu_dram_simulated` | 256 | 8 | 42 | synthetic |
| `real_world_incremental_pagerank` | ≤512 (padded to 16×) | real edges (85%/15% split) | — | `src/tests/cit-HepPh.txt` |
| `real_world_web_google_pagerank` | ≤512 (padded to 16×) | real edges (85%/15% split) | — | `src/tests/web-Google.txt` |
| `real_world_roadnet_pagerank` | ≤512 (padded to 16×) | real edges (85%/15% split) | — | `src/tests/roadNet-CA.txt` |

To change graph size or density for synthetic tests, edit the `const int N` and `avg_deg` values at the top of each test in [src/tests/PageRankTestCases.cpp](src/tests/PageRankTestCases.cpp), then rebuild with `scons`. For real-world tests, change `MAX_NODES` (must be a multiple of 16, or set `N % 16` padding applies automatically).
