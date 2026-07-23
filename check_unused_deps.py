"""One-off script: imports the 3 wrappers (triggers their sys.path.insert +
import chains for mediQ/agentclinic/meddxagent), then diffs sys.modules
against requirements.txt to flag declared-but-never-imported packages.

Run from repo root, inside the venv being audited:
    python check_unused_deps.py

Note: this only catches import-time usage. A package imported deep inside a
runtime-only branch we never exercise (e.g. an unused RAG code path) won't
show up as "used" even if some feature would need it — cross-check against
the -T limit=1 validation runs, which exercise real code, not just imports.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "wrapped_inspect"))

before = set(sys.modules.keys())

import inspect_agentclinic_wrapped  # noqa: F401
import inspect_mediq_wrapped  # noqa: F401
import inspect_meddxagent_wrapped  # noqa: F401

after = set(sys.modules.keys())
imported_top_level = {m.split(".")[0].lower().replace("_", "-") for m in (after - before)}

req_path = Path(__file__).parent / "requirements.txt"
declared = set()
for line in req_path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name = re.split(r"[<>=!\[]", line)[0].strip().lower()
    declared.add(name)

# a few PyPI-name -> import-name mismatches worth normalizing
ALIASES = {
    "faiss-cpu": "faiss",
    "beautifulsoup4": "bs4",
    "pyyaml": "yaml",
    "inspect-ai": "inspect-ai",  # imports as inspect_ai, already normalized above
    "python-dotenv": "dotenv",
    "sentence-transformers": "sentence-transformers",
    "langchain-openai": "langchain-openai",
}

unused = []
for pkg in sorted(declared):
    import_name = ALIASES.get(pkg, pkg)
    if import_name not in imported_top_level and pkg not in imported_top_level:
        unused.append(pkg)

print("Declared in requirements.txt but NOT seen in sys.modules after importing all 3 wrappers:")
for pkg in unused:
    print(" -", pkg)
if not unused:
    print(" (none — every declared package was imported)")
