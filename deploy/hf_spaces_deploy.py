import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kompress.tools.hf_deploy import main

if __name__ == "__main__":
    main()
