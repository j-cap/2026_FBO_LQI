"""sys.path bootstrap so `import src...` resolves to this project's own package
regardless of pytest's invocation directory (mirrors paper_CFL_FF/tests/conftest.py).
Must run before any `import src...` elsewhere in the test suite, and before
`src.ifac_bridge` appends the sibling IFAC project's root to sys.path."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
