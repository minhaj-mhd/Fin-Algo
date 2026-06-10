import os

def gauntlet_root() -> str:
    """
    Returns the root directory for gauntlet run directories, ledger, and caches.
    Can be overridden via the GAUNTLET_ROOT environment variable.
    """
    return os.environ.get("GAUNTLET_ROOT", "data/gauntlet")
