"""Few-video object-state tracking on MOSCATO (VidOSC-based approaches).

Shared modules live here; each approach is a subpackage with its own `run.py`:
states, attributes, embedding, discovery.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
