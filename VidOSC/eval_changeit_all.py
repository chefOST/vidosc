"""Evaluate every released ChangeIt checkpoint and track results in Weights & Biases.

Each checkpoint is evaluated in its own `train_changeit.py` process (one W&B run each,
grouped by setting); a final summary run aggregates the headline numbers.
Completed evals are skipped on re-run (based on eval_metrics.json).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time


# Reference numbers from the VidOSC paper (arXiv 2312.11782v2), Table 2 and Table 7.
_OW = ["open_world/known/state_prec@1", "open_world/novel/state_prec@1",
       "open_world/known/action_prec@1", "open_world/novel/action_prec@1"]
PAPER_SINGLE = {"single_task/state_prec@1": 0.57, "single_task/action_prec@1": 0.84,
                **dict(zip(_OW, [0.56, 0.48, 0.89, 0.82]))}
PAPER_MULTI = {"multi_task/state_prec@1": 0.44, "multi_task/action_prec@1": 0.69,
               **dict(zip(_OW, [0.43, 0.29, 0.75, 0.63]))}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", default="./checkpoints_changeit")
    p.add_argument("--log_root", default="./logs/changeit/eval")
    p.add_argument("--wandb_project", default="vidosc - full evals")
    p.add_argument("--wandb_entity", default="chefOST")
    p.add_argument("--accelerator", default="cpu")
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--settings", nargs="*", default=["single_task", "multi_task", "open_world"])
    p.add_argument("--categories", nargs="*", default=None, help="restrict single-task categories")
    p.add_argument("--rerun", action="store_true", help="ignore cached eval_metrics.json")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--no_summary", action="store_true", help="skip the W&B summary run")
    return p.parse_args()


def build_jobs(args):
    jobs = []
    if "single_task" in args.settings:
        for cat_dir in sorted(glob.glob(os.path.join(args.ckpt_root, "single_task", "*"))):
            cat = os.path.basename(cat_dir)
            if args.categories and cat not in args.categories:
                continue
            for kind in ["best_state", "best_action"]:
                ckpts = glob.glob(os.path.join(cat_dir, kind, "*", "*.ckpt"))
                assert len(ckpts) == 1, ckpts
                jobs.append(dict(setting="single_task", category=cat, kind=kind, ckpt=ckpts[0],
                                 flags=["--det", "0", "--category", cat]))
    for setting, extra in [("multi_task", []), ("open_world", ["--novel_obj"])]:
        if setting not in args.settings:
            continue
        for kind in ["state", "action"]:
            jobs.append(dict(setting=setting, category=None, kind=f"best_{kind}",
                             ckpt=os.path.join(args.ckpt_root, setting, f"{kind}_best.ckpt"),
                             flags=["--det", "1", "--use_gt_action", "--ordering"] + extra))
    for j in jobs:
        j["name"] = "/".join(x for x in [j["setting"], j["category"], j["kind"]] if x)
    return jobs


def run_job(args, job):
    log_name = os.path.relpath(os.path.join(args.log_root, job["name"]), "./logs/changeit")
    out_dir = os.path.join("./logs/changeit", log_name)
    metrics_file = os.path.join(out_dir, "eval_metrics.json")
    if os.path.exists(metrics_file) and not args.rerun:
        print(f"[skip] {job['name']} (cached)")
        return json.load(open(metrics_file))
    cmd = [sys.executable, "train_changeit.py", *job["flags"], "--ckpt", job["ckpt"],
           "--log_name", log_name, "--accelerator", args.accelerator, "--num_workers", str(args.num_workers),
           "--wandb_project", args.wandb_project, "--wandb_group", job["setting"], "--wandb_name", job["name"],
           "--wandb_tags", job["setting"], job["kind"], *([f"category:{job['category']}"] if job["category"] else [])]
    if args.wandb_entity:
        cmd += ["--wandb_entity", args.wandb_entity]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"[run ] {job['name']}", flush=True)
    with open(os.path.join(out_dir, "stdout.log"), "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(metrics_file):
        print(f"[FAIL] {job['name']} rc={rc}, see {out_dir}/stdout.log", flush=True)
        return None
    m = json.load(open(metrics_file))
    print(f"[done] {job['name']} {time.time() - t0:.0f}s  state={m.get('val_avg_prec_1', float('nan')):.4f} "
          f"action={m.get('val_avg_action_prec_1', float('nan')):.4f}", flush=True)
    return m


def summarize(args, jobs, results):
    import wandb

    def get(name, key):
        r = results.get(name)
        return None if r is None or key not in r else float(r[key])

    summary, tables = {}, {}
    single = [j for j in jobs if j["setting"] == "single_task" and j["kind"] == "best_state"]
    if single:
        rows = []
        for j in single:
            cat = j["category"]
            rows.append([cat, get(f"single_task/{cat}/best_state", "val_avg_prec_1"),
                         get(f"single_task/{cat}/best_action", "val_avg_action_prec_1"),
                         get(f"single_task/{cat}/best_state", "val_avg_action_prec_1"),
                         get(f"single_task/{cat}/best_action", "val_avg_prec_1")])
        tables["single_task_per_category"] = wandb.Table(
            columns=["category", "state_prec@1 (best_state ckpt)", "action_prec@1 (best_action ckpt)",
                     "action_prec@1 (best_state ckpt)", "state_prec@1 (best_action ckpt)"], data=rows)
        for col, idx in [("state_prec@1", 1), ("action_prec@1", 2)]:
            vals = [r[idx] for r in rows if r[idx] is not None]
            summary[f"single_task/{col}"] = sum(vals) / len(vals) if vals else None
            summary[f"single_task/n_categories_{col}"] = len(vals)

    for setting in ["multi_task", "open_world"]:
        if setting not in args.settings:
            continue
        s, a = f"{setting}/best_state", f"{setting}/best_action"
        summary[f"{setting}/state_prec@1"] = get(s, "val_avg_prec_1")
        summary[f"{setting}/action_prec@1"] = get(a, "val_avg_action_prec_1")
        if setting == "open_world":
            summary["open_world/known/state_prec@1"] = summary.pop("open_world/state_prec@1")
            summary["open_world/known/action_prec@1"] = summary.pop("open_world/action_prec@1")
            summary["open_world/novel/state_prec@1"] = get(s, "test_avg_prec_1")
            summary["open_world/novel/action_prec@1"] = get(a, "test_avg_action_prec_1")

    headline = [[k, v, PAPER_SINGLE.get(k), PAPER_MULTI.get(k)]
                for k, v in summary.items() if "n_categories" not in k]
    tables["headline"] = wandb.Table(
        columns=["metric", "reproduced", "paper (single-task, Tab. 2)", "paper (multi-task, Tab. 7)"], data=headline)
    all_rows = [[n, *(results[n].get(k) if results.get(n) else None for k in
                      ["val_avg_prec_1", "val_avg_action_prec_1", "test_avg_prec_1", "test_avg_action_prec_1"])]
                for n in [j["name"] for j in jobs]]
    tables["all_checkpoints"] = wandb.Table(
        columns=["run", "val_avg_prec_1", "val_avg_action_prec_1", "test_avg_prec_1", "test_avg_action_prec_1"],
        data=all_rows)

    # Multi-task: one bar chart per checkpoint with categories on the x-axis, plus reproduced vs. paper.
    charts = {}
    ms, ma = results.get("multi_task/best_state"), results.get("multi_task/best_action")
    if ms and ma:
        cats = sorted(k[len("val_prec_1_"):] for k in ms if k.startswith("val_prec_1_"))
        rows = [[c, ms[f"val_prec_1_{c}"], ma[f"val_action_prec_1_{c}"]] for c in cats]
        tables["multi_task_per_category"] = wandb.Table(
            columns=["category", "state_prec@1 (state_best ckpt)", "action_prec@1 (action_best ckpt)"], data=rows)
        for col, idx, ckpt in [("state_prec@1", 1, "state_best"), ("action_prec@1", 2, "action_best")]:
            t = wandb.Table(columns=["category", col], data=sorted([[r[0], r[idx]] for r in rows], key=lambda r: -r[1]))
            charts[f"multi_task/{col}_per_category"] = wandb.plot.bar(
                t, "category", col, title=f"Multi-task {col} per category ({ckpt}.ckpt)")
        vs = [[f"{m} ({src})", val] for m in ["state_prec@1", "action_prec@1"]
              for src, val in [("reproduced", summary.get(f"multi_task/{m}")), ("paper", PAPER_MULTI[f"multi_task/{m}"])]]
        charts["multi_task/reproduced_vs_paper"] = wandb.plot.bar(
            wandb.Table(columns=["metric", "value"], data=vs), "metric", "value",
            title="Multi-task: reproduced vs. paper (Table 7)")

    name = "summary" if len(args.settings) == 3 else "summary/" + "+".join(args.settings)
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=name, group="summary",
                     job_type="summary", config={"settings": args.settings, "n_jobs": len(jobs)})
    run.log({**tables, **charts})
    run.summary.update({k: v for k, v in summary.items() if v is not None})
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k:45s} {v if v is None or isinstance(v, int) else f'{v:.4f}'}")
    print(f"W&B: {run.url}")
    run.finish()
    with open(os.path.join(args.log_root, name.replace("/", "_") + ".json"), "w") as f:
        json.dump({"summary": summary, "per_run": results}, f, indent=4)


def main():
    args = parse_args()
    jobs = build_jobs(args)
    print(f"{len(jobs)} eval jobs")
    os.makedirs(args.log_root, exist_ok=True)
    results = {}
    for job in jobs:
        if args.summary_only:
            f = os.path.join(args.log_root, job["name"], "eval_metrics.json")
            results[job["name"]] = json.load(open(f)) if os.path.exists(f) else None
        else:
            results[job["name"]] = run_job(args, job)
    failed = [n for n, r in results.items() if r is None]
    if failed:
        print(f"{len(failed)} failed/missing: {failed}")
    if not args.no_summary:
        summarize(args, jobs, results)


if __name__ == "__main__":
    main()
