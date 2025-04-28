import os
import re
import json
import pandas as pd

LOG_DIR = "logs"

def parse_filename_old(fname):
    
    base = os.path.basename(fname)
    m = re.match(r"(.+)_results_(angular|random)_n?([0-9.]*)t?([0-9.]*)", base)
    if not m:
        return None, None, None
    dataset = m.group(1)
    approach = m.group(2)
    n = m.group(3)
    t = m.group(4)
    param = ""
    if approach == "angular":
        param = f"n={n}"
    elif approach == "random":
        param = f"t={t}"
    return dataset, approach, param

def parse_filename(fname):
    
    base = os.path.basename(fname)
    
    m = re.match(r"(.+)_results_(angular|random)_n?([0-9.]*)t?([0-9.]*)", base)
    if m:
        dataset = m.group(1)
        approach = m.group(2)
        n = m.group(3)
        t = m.group(4)
        param = ""
        if approach == "angular":
            param = f"n={n}"
        elif approach == "random":
            param = f"t={t}"
        return dataset, approach, param
    
    m = re.match(r"(.+)_results_base(_\d+)?", base)
    if m:
        dataset = m.group(1)
        approach = "base"
        param = ""
        return dataset, approach, param
    return None, None, None

def extract_last_json(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
    
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line.replace("'", '"'))
            except Exception:
                continue
    return None

results = []
for fname in os.listdir(LOG_DIR):
    path = os.path.join(LOG_DIR, fname)
    if not os.path.isfile(path):
        continue
    dataset, approach, param = parse_filename(fname)
    if not dataset:
        continue
    metrics = extract_last_json(path)
    if not metrics:
        continue
    row = {
        "dataset": dataset,
        "approach": approach,
        "param": param,
    }
    
    for k, v in metrics.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                row[f"{k}_{subk}"] = subv
        else:
            row[k] = v
    results.append(row)

df = pd.DataFrame(results)


df['approach'] = df['approach'].replace({'random': 'random_dropout'})
approach_order = pd.CategoricalDtype(['base', 'random_dropout', 'angular'], ordered=True)
df['approach'] = df['approach'].astype(approach_order)

# Table 1: Metrics
metric_cols = [
    "dataset", "approach", "param",
    "predicted_text_rouge-l", "predicted_text_rouge-1", "predicted_text_rouge-2",
    "predicted_text_bleu_score", "predicted_text_exact_match", "acceptance_rate_mean"
]
df_metrics = df[metric_cols].copy()
df_metrics = df_metrics.sort_values(by=["dataset", "approach", "param"])
df_metrics.to_csv("summary_metrics.csv", index=False)
print("\n=== METRICS TABLE ===")
print(df_metrics.to_markdown(index=False))

# Table 2: Timing
timing_cols = [
    "dataset", "approach", "param",
    "total_time_mean", "time_per_token_mean", "tokens_per_second_mean"
]
df_timing = df[timing_cols].copy()
df_timing = df_timing.sort_values(by=["dataset", "approach", "param"])
df_timing.to_csv("summary_timing.csv", index=False)
print("\n=== TIMING TABLE ===")
print(df_timing.to_markdown(index=False))