import sys
from pathlib import Path

# Make the golden corpus package (tests/adapters/golden) importable from the
# engine test directory, mirroring pytest's rootdir insertion for
# tests/adapters itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
