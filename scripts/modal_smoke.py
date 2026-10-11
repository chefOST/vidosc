"""Check Modal authentication and remote CPU execution.

Run from the repository root:
    .venv/bin/python -m modal run scripts/modal_smoke.py
"""

import modal

app = modal.App("vidosc-connection-check")


@app.function(timeout=60, retries=0, max_containers=1)
def square(x: int) -> int:
    print("Running on a Modal remote CPU worker.")
    return x**2


@app.local_entrypoint()
def main():
    result = square.remote(42)
    if result != 1764:
        raise RuntimeError(f"Unexpected result: {result}")
    print(f"Modal connection verified: 42 squared = {result}")
