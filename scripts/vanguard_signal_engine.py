import os
import sys

# Add project root to path before importing local modules
sys.path.append(os.getcwd())

from scripts.vanguard.orchestrator import VanguardOrchestrator
from scripts.terminal_utils import log

class VanguardEngine(VanguardOrchestrator):
    """
    VanguardEngine compatibility wrapper.
    Inherits all core capabilities from VanguardOrchestrator to preserve 
    backward compatibility with batch startup scripts and local modules.
    """
    def __init__(self, model_path=None, scaler_path=None, meta_path=None):
        super().__init__(model_path, scaler_path, meta_path)

if __name__ == "__main__":
    try:
        engine = VanguardEngine()
        engine.run()
    except KeyboardInterrupt:
        log("\n[SHUTDOWN] Vanguard Engine terminated by operator (Ctrl+C).")
    except Exception as e:
        log(f"\n[FATAL ERROR] Engine crashed: {e}")
        import traceback
        traceback.print_exc()
