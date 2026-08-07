"""Staged continual pre-training over M2D2 domains.

Layout
------
``data/tokenize_pack.py``  raw ``.txt`` -> ``uint32`` packed memmaps + ``meta.json``
``data/loader.py``         packed memmaps -> torch ``Dataset``, plus replay mixing
``train/cpt.py``           sequential per-domain stage runner (entry point)
``train/sft.py``           deliberate stub for the downstream SFT phase
``eval/perplexity.py``     per-domain held-out NLL
``eval/matrix.py``         ``ppl_matrix.jsonl`` -> stage x domain table + CL metrics
"""

__version__ = "0.1.0"