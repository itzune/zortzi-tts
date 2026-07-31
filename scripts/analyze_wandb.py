#!/usr/bin/env python3
"""Pull prosody + training metrics from the latest wandb run."""
import os
import collections

import wandb

# Load wandb token
with open("/root/work/.env") as f:
    for line in f:
        if line.startswith("WANDB_TOKEN="):
            os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip()

api = wandb.Api()

# Find the latest CV base run
runs = api.runs("itzune/zortzi-tts", order="-created_at", per_page=10)
cv_run = runs[0]  # latest

print("Run: %s (%s)" % (cv_run.name, cv_run.id))
print("State: %s" % cv_run.state)
print()

# Pull full history (scan_rows for all logged steps)
history = list(cv_run.scan_history())

# ---- Prosody metrics ----
prosody_keys = set()
for row in history:
    for k in row.keys():
        if k.startswith("prosody/"):
            prosody_keys.add(k)

print("Prosody metrics found: %d keys" % len(prosody_keys))

# Extract step + prosody values
steps_data = collections.defaultdict(dict)
for row in history:
    step = row.get("_step", None)
    if step is None:
        continue
    for k, v in row.items():
        if k.startswith("prosody/") and v is not None:
            steps_data[step][k] = v

print("\n=== PROSODY TRENDS (by checkpoint step) ===")
for step in sorted(steps_data.keys()):
    data = steps_data[step]
    if not data:
        continue
    print("\n--- step %d ---" % step)
    for voice in ["default", "maider"]:
        for sidx in [1, 2]:
            prefix = "prosody/%s/s%d/" % (voice, sidx)
            metrics = {k.replace(prefix, ""): v for k, v in data.items() if k.startswith(prefix)}
            if metrics:
                stype = "decl" if sidx == 1 else "wh-Q"
                f0std = metrics.get("f0_std_hz", "?")
                fpeak = metrics.get("focus_peak_hz", "-")
                bdelta = metrics.get("boundary_delta_hz", "?")
                fmean = metrics.get("f0_mean_hz", "?")
                print("  %-8s s%d(%s): f0_std=%-6s focus_peak=%-6s bndry_d=%-6s mean=%s" %
                      (voice, sidx, stype, f0std, fpeak, bdelta, fmean))

# ---- Training loss/accuracy ----
print("\n=== TRAINING METRICS (sampled) ===")
train_steps = []
for row in history:
    if "loss" in row:
        train_steps.append(row)
if train_steps:
    n = len(train_steps)
    # sample ~8 evenly across training
    idxs = [int(i * (n - 1) / 7) for i in range(8)]
    print("total logged steps: %d" % n)
    for i in idxs:
        r = train_steps[i]
        step = r.get("_step", "?")
        print("  step %-5s loss=%.2f slow=%.2f fast=%.2f slow_acc=%.3f fast_acc=%.3f lr=%.2e" %
              (step, r.get("loss", 0), r.get("slow_loss", 0), r.get("fast_loss", 0),
               r.get("slow_accuracy", 0), r.get("fast_accuracy", 0), r.get("learning_rate", 0)))
