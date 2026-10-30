"""Plan-first sequential VidOSC runner. Execution requires explicit flags.

Completed stages are reused only when their command, inputs, and outputs match.
No downloads or W&B uploads are performed. No automatic retries of paid jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_stage(stage, execute=False, allow_modal=False):
    receipt = Path(stage["receipt"])
    print(stage["name"], json.dumps(stage["command"]), flush=True)
    if not execute:
        return "planned"
    if stage.get("modal") and not allow_modal:
        raise ValueError("Modal execution requires --allow-modal after budget approval")
    identity = {"command": stage["command"], "env": stage.get("env", {}),
                "inputs": {p: fingerprint(p) for p in stage["inputs"]}}
    if receipt.exists():
        old = json.loads(receipt.read_text())
        actual = {p: fingerprint(p) for p in stage["outputs"]}
        if old["identity"] != identity or old["outputs"] != actual:
            raise ValueError("Changed inputs or outputs; use a new experiment directory")
        print("  Reusing verified completed stage", flush=True)
        return "reused"
    if any(Path(p).exists() for p in stage["outputs"]):
        raise ValueError("Unverified partial outputs exist; inspect and move them before retrying")
    subprocess.run(stage["command"], cwd=ROOT,
                   env={**os.environ, **stage.get("env", {})}, check=True)
    result = {"identity": identity,
              "outputs": {p: fingerprint(p) for p in stage["outputs"]}}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(receipt)
    return "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-modal", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=1)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.max_jobs < 1 or len(plan["jobs"]) > args.max_jobs:
        parser.error("Job count exceeds --max-jobs; explicitly choose the intended batch size")
    # Check all gates before any local feature extraction starts.
    if args.execute and not args.allow_modal and any(s.get("modal") for j in plan["jobs"] for s in j["stages"]):
        parser.error("This plan includes Modal; execution requires --allow-modal")
    for job in plan["jobs"]:
        print(f"Job: {job['name']}")
        for stage in job["stages"]:
            run_stage(stage, args.execute, args.allow_modal)
    if not args.execute:
        print("DRY RUN ONLY: no models, remote jobs, or uploads were started.")


if __name__ == "__main__":
    main()
