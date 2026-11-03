# Multi-Modal Federated Learning with Agentic Computation Offloading

A research prototype demonstrating federated learning across four data modalities
(image, text, audio, video) with an agentic controller for smart client selection
and computation offloading.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (runs offline, no internet needed)
python synthetic_data.py --num_clients 50 --alpha 0.5

# 3. Run all phases in order
python -m phase1_baselines.train_baselines      # Verify models learn
python -m phase2_fl_sim.run_fl_simulation        # FedAvg FL per modality
python -m phase3_router.test_router              # Test modality router
python -m phase4_offload.simulate_offload        # Offload decision simulation
python -m phase5_agentic.run_comparison          # Random vs agentic selection
python -m phase6_evaluation.generate_report      # Generate all plots/reports

# 4. Run full Results Extraction Suite (generates all 22 CSVs and 17 PNG plots)
python run_results_extraction.py

# 5. (Optional) Run on real data (downloads CIFAR-10 ~170MB)
python -m phase7_real_data.run_real_data --dataset cifar10
python -m phase7_real_data.compare_synthetic_vs_real
```

## Architecture

```
Cloud (Orchestrator)        Edge (Simulated)         Mobile (Clients)
+-------------------+      +----------------+       +----------------+
| Agentic Controller|      | Edge Server    |       | 50 Image Phones|
|  - Client scoring |      |  (powerful CPU)|       | 50 Text Phones |
|  - Offload policy |      |  Runs heavy    |       | 50 Audio Phones|
|                   |      |  training jobs |       | 50 Video Phones|
| FedAvg Aggregators|      |  (video/audio) |       |                |
|  x4 (one/modality)|      +--------+-------+       +-------+--------+
+---------+---------+               |                        |
          |                         |                        |
          +-------------------------+------------------------+
                    Model weights flow (never raw data)
```

## Project Structure

```
Multimodel FL Agentic AI/
|-- requirements.txt              # Dependencies
|-- partition_utils.py            # Dirichlet non-IID partitioning
|-- synthetic_data.py             # Synthetic data generator
|-- torch_dataset_wrappers.py     # PyTorch Dataset/DataLoader wrappers
|-- real_data_loaders.py          # CIFAR-10, AG News, etc.
|-- synthetic_fl_data/            # Generated data shards
|
|-- phase1_baselines/             # Phase 1: Model validation
|   |-- models.py                 # ImageCNN, TextLSTM, AudioCNN, VideoCNN
|   |-- train_baselines.py        # Centralized training
|   +-- results/
|
|-- phase2_fl_sim/                # Phase 2: FL simulation
|   |-- fl_utils.py               # FedAvg, param helpers
|   |-- run_fl_simulation.py      # Per-modality FL
|   +-- results/
|
|-- phase3_router/                # Phase 3: File routing
|   |-- modality_router.py        # Extension/MIME routing
|   +-- test_router.py            # 100-file test
|
|-- phase4_offload/               # Phase 4: Offload decisions
|   |-- offload_decision.py       # Cost function
|   |-- simulate_offload.py       # Multi-round simulation
|   +-- results/
|
|-- phase5_agentic/               # Phase 5: Smart selection
|   |-- client_scorer.py          # 5-feature scoring
|   |-- agentic_strategy.py       # AgenticFedAvg
|   |-- run_comparison.py         # Random vs agentic
|   +-- results/
|
|-- phase6_evaluation/            # Phase 6: Reports
|   |-- generate_report.py        # Dashboard + JSON
|   +-- results/
|
+-- phase7_real_data/             # Phase 7: Real datasets
    |-- run_real_data.py          # FL on real data
    |-- compare_synthetic_vs_real.py
    +-- results/
```

## Models

| Modality | Model       | Parameters | Input Shape          | Classes |
|----------|-------------|-----------|----------------------|---------|
| Image    | ImageCNN    | ~620K     | (B, 3, 32, 32)      | 10      |
| Text     | TextLSTM    | ~550K     | (B, 50) token IDs   | 4       |
| Audio    | AudioCNN    | ~620K     | (B, 1, 40, 100)     | 10      |
| Video    | VideoCNN    | ~160K     | (B, 8, 3, 32, 32)   | 5       |

## Key Results

### Phase 1: All models learn above chance on synthetic data
- Image: 100%, Text: 96.75%, Audio: 100%, Video: 100%

### Phase 3: 100% modality routing accuracy

### Phase 4: Offload decisions match expectations
- Video: 43.7% offloaded, Audio: 7.5%, Image: 3.6%, Text: 1.7%
- Heavy modalities offload more than light ones

## Configuration

- **Clients per modality**: 50 (configurable via `--num_clients`)
- **FL rounds**: 20 (configurable via `--rounds`)
- **Dirichlet alpha**: 0.5 (controls non-IID level)
- **Client selection**: Top 60% by composite score
- **Local epochs**: 1 per round
- **Learning rate**: 0.001

## Tech Stack

- Python 3.10+
- PyTorch >= 2.0
- NumPy, matplotlib, scikit-learn
- No internet required for Phases 1-6 (synthetic data only)
