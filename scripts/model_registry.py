"""
model_registry.py - Model Registry for Vanguard XGBoost Models

Usage (CLI):
  python scripts/model_registry.py --list
  python scripts/model_registry.py --active
  python scripts/model_registry.py --switch v2_feature_fix
  python scripts/model_registry.py --switch v2_feature_fix --dry-run

Usage (in code):
  from scripts.model_registry import ModelRegistry
  registry = ModelRegistry()
  active = registry.get_active_model()
  long_path  = active["long_model"]
  short_path = active["short_model"]
  meta_path  = active["meta"]
"""

import os
import sys
import json
import argparse
from datetime import datetime

REGISTRY_PATH = "models/registry.json"


class ModelRegistry:

    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.registry_path = registry_path
        self._registry = self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(
                f"Registry not found at '{self.registry_path}'. "
                "Run the setup script or create models/registry.json."
            )
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _save(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def get_active_model(self) -> dict:
        """Return the full config dict for the currently active model."""
        active_name = self._registry.get("active_model")
        if not active_name:
            raise ValueError("No 'active_model' set in registry.json")
        model = self._registry["models"].get(active_name)
        if not model:
            raise ValueError(f"Active model '{active_name}' not found in registry")
        return {**model, "name": active_name}

    def get_active_name(self) -> str:
        return self._registry.get("active_model", "unknown")

    def list_models(self) -> list:
        """Return list of all model dicts including their names."""
        return [
            {**v, "name": k}
            for k, v in self._registry.get("models", {}).items()
        ]

    def switch_model(self, model_name: str, dry_run: bool = False) -> bool:
        """
        Switch the active model to model_name.

        Validates that all required files exist before switching.
        If dry_run=True, only validates without making any changes.

        Returns True on success, False on failure.
        """
        models = self._registry.get("models", {})
        if model_name not in models:
            print(f"[ERROR] Model '{model_name}' not found in registry.")
            print(f"  Available: {list(models.keys())}")
            return False

        model = models[model_name]
        required = ["long_model", "short_model", "scaler", "meta"]
        missing  = []

        for key in required:
            path = model.get(key)
            # Scaler is optional — an empty string means the model is scaler-free
            if key == "scaler" and not path:
                continue
            if not path or not os.path.exists(path):
                missing.append(f"  {key}: {path or 'not set'}")

        if missing:
            print(f"[ERROR] Cannot switch to '{model_name}' — missing files:")
            for m in missing:
                print(m)
            print(f"\n  Train the model first:")
            print(f"    python scripts/train_ranking_v2.py")
            return False

        current = self._registry.get("active_model", "unknown")

        if dry_run:
            print(f"[DRY RUN] Would switch: '{current}' -> '{model_name}'")
            print(f"  All required files exist. Switch is safe.")
            return True

        self._registry["active_model"] = model_name
        self._save()
        print(f"[OK] Model switched: '{current}' -> '{model_name}'")
        print(f"  Long  : {model['long_model']}")
        print(f"  Short : {model['short_model']}")
        print(f"  Meta  : {model['meta']}")
        print(f"\n  Restart the Vanguard engine to apply the new model.")
        return True

    def register_model(self, name: str, config: dict) -> None:
        """
        Add or update a model entry in the registry.

        config should contain:
          description, long_model, short_model, scaler, meta,
          type, long_test_spearman, short_test_spearman, notes
        """
        config["trained_at"] = config.get("trained_at", datetime.now().isoformat())
        self._registry["models"][name] = config
        self._save()
        print(f"[OK] Registered model '{name}' in registry.")

    def print_status(self) -> None:
        """Print a formatted table of all models and highlight the active one."""
        active = self._registry.get("active_model", "none")
        models = self._registry.get("models", {})

        print("\n" + "=" * 75)
        print("VANGUARD MODEL REGISTRY")
        print("=" * 75)
        print(f"{'Name':<25} {'Type':<10} {'Long rho':>8} {'Short rho':>8} {'Trained':>12} {'Active':>7}")
        print("-" * 75)

        for name, m in models.items():
            long_rho  = m.get("long_test_spearman")
            short_rho = m.get("short_test_spearman")
            trained   = m.get("trained_at") or "Not yet"
            if trained and trained != "Not yet":
                try:
                    trained = trained[:10]  # date only
                except Exception:
                    pass
            is_active = "* YES" if name == active else ""

            print(
                f"{'  ' + name if name == active else name:<25}"
                f" {m.get('type','?'):<10}"
                f" {f'{long_rho:.4f}' if long_rho is not None else 'N/A':>8}"
                f" {f'{short_rho:.4f}' if short_rho is not None else 'N/A':>8}"
                f" {trained:>12}"
                f" {is_active:>7}"
            )

        print("=" * 75)
        print(f"Active: {active}")
        print()

        active_model = models.get(active, {})
        print(f"  Long  model : {active_model.get('long_model', 'N/A')}")
        print(f"  Short model : {active_model.get('short_model', 'N/A')}")
        print(f"  Metadata    : {active_model.get('meta', 'N/A')}")
        print(f"  Description : {active_model.get('description', 'N/A')}")
        print("=" * 75 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vanguard Model Registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/model_registry.py --list
  python scripts/model_registry.py --active
  python scripts/model_registry.py --switch v2_feature_fix
  python scripts/model_registry.py --switch v2_feature_fix --dry-run
        """
    )
    parser.add_argument("--list",      action="store_true", help="List all registered models")
    parser.add_argument("--active",    action="store_true", help="Show active model info")
    parser.add_argument("--switch",    type=str,            help="Switch to named model")
    parser.add_argument("--dry-run",   action="store_true", help="Validate switch without applying")

    args = parser.parse_args()

    try:
        registry = ModelRegistry()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.list:
        registry.print_status()

    elif args.active:
        m = registry.get_active_model()
        print(f"\nActive model: {m['name']}")
        print(f"  Long  : {m['long_model']}")
        print(f"  Short : {m['short_model']}")
        print(f"  Meta  : {m['meta']}")
        print(f"  Long rho  (test): {m.get('long_test_spearman')}")
        print(f"  Short rho (test): {m.get('short_test_spearman')}")

    elif args.switch:
        registry.switch_model(args.switch, dry_run=args.dry_run)

    else:
        registry.print_status()
