from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[2]
FOLDWISE_IMPLEMENTATION = CODE_ROOT / "foldwise_experiment"

if str(FOLDWISE_IMPLEMENTATION) not in sys.path:
    sys.path.insert(0, str(FOLDWISE_IMPLEMENTATION))

from run_within_project import main


if __name__ == "__main__":
    main()
