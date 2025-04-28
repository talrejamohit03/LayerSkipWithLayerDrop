# Experiments using LayerDrop and Layer Pruning on top of LayerSkip

This repository extends the [LayerSkip](https://arxiv.org/abs/2404.16710) codebase with **layer pruning experiments** using two approaches: **Angular Distance Pruning** and **Randomized Dropout Pruning**. The goal is to benchmark the impact of structured and random pruning on model accuracy and inference speed.

---

## Key Changes

### 1. `benchmark.py` Enhancements

- **New Command-Line Arguments**:  
  - `--prune_method`: Selects pruning strategy (`angular`, `random`, or `base`).
  - `--prune_n`: Number of layers to prune (for angular).
  - `--dropout_threshold`: Dropout probability (for random).
- **Pruning Integration**:  
  - The script initializes a `PruneModel` and applies the selected pruning method before benchmarking.
  - Results (metrics and timing) are saved for each run.
- **Flexible Experimentation**:  
  - Easily sweep over datasets, pruning methods, and parameters via shell scripts.

### 2. `pruned_model.py` Additions

- **`PruneModel` Class**:  
  - Implements both pruning strategies:
    - **Angular Distance Pruning**: Identifies and prunes `n` consecutive layers with minimal angular distance between activations.
    - **Randomized Dropout Pruning**: Randomly prunes layers with probability `dropout_threshold`.
  - Handles tokenizer setup, dataloader creation, and layer manipulation.
  - Provides hooks for verifying which layers were pruned.

---

## How to Run Benchmarks

### 1. Install Dependencies

```bash
conda create --name layer_skip python=3.10
conda activate layer_skip
pip install -r requirements.txt
```

### 2. Run Experiments

#### Example: Angular Distance Pruning

```bash
torchrun benchmark.py --model facebook/layerskip-llama2-7B \
    --dataset cnn_dm_summarization \
    --num_samples 20 \
    --generation_strategy self_speculative \
    --exit_layer 30 \
    --num_speculations 6 \
    --output_dir ./logs/eval/ \
    --prune_method angular \
    --prune_n 2
```

#### Example: Randomized Dropout Pruning

```bash
torchrun benchmark.py --model facebook/layerskip-llama2-7B \
    --dataset cnn_dm_summarization \
    --num_samples 20 \
    --generation_strategy self_speculative \
    --exit_layer 30 \
    --num_speculations 6 \
    --output_dir ./logs/eval/ \
    --prune_method random \
    --dropout_threshold 0.2
```

#### Example: Baseline (No Pruning)

```bash
torchrun benchmark.py --model facebook/layerskip-llama2-7B \
    --dataset cnn_dm_summarization \
    --num_samples 20 \
    --generation_strategy self_speculative \
    --exit_layer 30 \
    --num_speculations 6 \
    --output_dir ./logs/eval/ \
    --prune_method base
```

### 3. Batch Experiments

Use the provided `run_tests` or `run_tests_base` scripts to sweep over all datasets and pruning parameters.

---

## Results Analysis

- Results are saved in the `logs/` directory.
- Use the `analyze_logs.py` script to aggregate results and generate summary tables (metrics and timing).
- The tables allow comparison of accuracy and speed across all pruning strategies and parameters.

---

## Code Structure

- `benchmark.py`: Main benchmarking script with pruning integration.
- `pruned_model.py`: Implements `PruneModel` and all pruning logic.
- `analyze_logs.py`: Aggregates and summarizes experiment results.
- `run_tests`, `run_tests_base`: Example scripts for running sweeps.

---

## Citation

If you use this code or results, please cite the original LayerSkip paper:

```bibtex
@misc{layerskip,
    title={LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding},
    author={Mostafa Elhoushi and Akshat Shrivastava and Diana Liskovich and Basil Hosmer and Bram Wasti and Liangzhen Lai and Anas Mahmoud and Bilge Acun and Saurabh Agarwal and Ahmed Roman and Ahmed A Aly and Beidi Chen and Carole-Jean Wu},
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.681",
    doi = "10.18653/v1/2024.acl-long.681",
    pages = "12622--12642",
}
```

---

## License

This project is licensed under CC-by-NC. See the LICENSE file for details.

---