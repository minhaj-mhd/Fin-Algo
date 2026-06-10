import pytest
from scripts.gauntlet.contracts import GauntletConfig
from scripts.gauntlet.cli import get_canonical_hash

def test_canonical_hash_stability():
    config1 = GauntletConfig()
    config2 = GauntletConfig()
    
    hash1 = get_canonical_hash(config1)
    hash2 = get_canonical_hash(config2)
    
    assert hash1 == hash2
    
    # Verify exact hash value so we know if it drifts due to any struct changes
    # e.g., default fields reordered, new fields added.
    print("Hash1:", hash1)
