import os
import time
import json
from google import genai
from google.genai import types

class GeminiRotator:
    def __init__(self, state_file="data/gemini_rotation_state.json"):
        self.state_file = state_file
        self.main_keys = []
        self.backup_key = None
        self.current_index = 0
        self.stats = {}
        # Hard per-request HTTP timeout (ms). Without this, an overloaded Gemini
        # endpoint that accepts the connection but never replies hangs the call —
        # and the whole scan loop — forever. Configurable via env for S2 (search)
        # calls which run longer than the S1 flash veto.
        try:
            self.request_timeout_ms = int(os.getenv("GEMINI_REQUEST_TIMEOUT_MS", "45000"))
        except (TypeError, ValueError):
            self.request_timeout_ms = 45000
        self.load_keys()
        self.load_state()

    def load_keys(self):
        # 1. Load Main Keys
        keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
        if keys_env:
            self.main_keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        else:
            self.main_keys = []
            
        # 2. Load Backup Key
        self.backup_key = os.getenv("BACKUP_GEMINI_API_KEY")
        if self.backup_key:
            self.backup_key = self.backup_key.strip()
            
    def load_state(self):
        self.current_index = 0
        self.stats = {}
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.current_index = int(data.get("current_index", 0))
                    self.stats = data.get("stats", {})
            except Exception as e:
                print(f"[GEMINI-ROTATOR] Warning: Failed to load state file: {e}")
        
        # Ensure rotation index is within bounds of currently configured main keys
        if self.main_keys:
            self.current_index = self.current_index % len(self.main_keys)
        else:
            self.current_index = 0

    def save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        try:
            with open(self.state_file, "w") as f:
                json.dump({"current_index": self.current_index, "stats": self.stats}, f, indent=4)
        except Exception as e:
            print(f"[GEMINI-ROTATOR] Warning: Failed to save rotation state: {e}")

    def execute(self, func, *args, **kwargs):
        """
        Executes a callable 'func(client, *args, **kwargs)' where client is a genai.Client.
        Always uses the current rotated main key first. If it encounters ANY error,
        it rotates to the next main key and tries again, until all main keys are exhausted.
        """
        # Reload keys in case of live environment changes
        self.load_keys()
        
        if not self.main_keys:
            raise ValueError("No Gemini API keys (main keys) are configured in the environment.")

        if not hasattr(self, 'key_available_time'):
            self.key_available_time = {}

        last_exception = None
        max_attempts = len(self.main_keys) * 3  # Try each key multiple times
        consecutive_503 = 0  # 503 is a MODEL-wide overload, not a per-key quota

        import re
        import random

        for attempt in range(max_attempts):
            now = time.time()
            best_index = self.current_index
            min_wait = float('inf')
            
            # Find the best key to use right now
            for i in range(len(self.main_keys)):
                idx = (self.current_index + i) % len(self.main_keys)
                key = self.main_keys[idx]
                available_at = self.key_available_time.get(key, 0)
                wait_time = max(0.0, available_at - now)
                
                if wait_time == 0:
                    best_index = idx
                    min_wait = 0.0
                    break
                elif wait_time < min_wait:
                    best_index = idx
                    min_wait = wait_time
            
            self.current_index = best_index
            
            if min_wait > 0:
                print(f"[GEMINI-ROTATOR] All keys on cooldown (min wait {min_wait:.2f}s). Failing fast to trigger model tier fallback.")
                break
                
            main_key = self.main_keys[self.current_index]
            next_index = (self.current_index + 1) % len(self.main_keys)
            
            key_id = main_key[:8] + "..." + main_key[-4:] if len(main_key) > 12 else "unknown"
            if key_id not in self.stats:
                self.stats[key_id] = {"success": 0, "fail": 0}
            
            try:
                print(f"[GEMINI-ROTATOR] Attempting request using main key {self.current_index + 1} of {len(self.main_keys)}")
                client = genai.Client(
                    api_key=main_key,
                    http_options=types.HttpOptions(timeout=self.request_timeout_ms),
                )
                result = func(client, *args, **kwargs)
                
                # Succeeded on main key. Rotate main index for the next request.
                self.stats[key_id]["success"] += 1
                self.current_index = next_index
                self.save_state()
                return result
                
            except Exception as e:
                e_str = str(e)
                # Print only the first line of the exception for cleaner logs
                first_line = e_str.splitlines()[0] if e_str.splitlines() else e_str
                print(f"[GEMINI-ROTATOR] Main key failed: {first_line}")
                
                self.stats[key_id]["fail"] += 1
                last_exception = e

                # Default inter-key delay (legitimate rate-limit backoff).
                backoff = random.uniform(3.0, 5.0)

                # Parse retry delay if it's a 429 RESOURCE_EXHAUSTED
                if "429" in e_str and "RESOURCE_EXHAUSTED" in e_str:
                    consecutive_503 = 0
                    match = re.search(r'Please retry in ([0-9.]+)s', e_str)
                    if match:
                        parsed_delay = float(match.group(1))
                        self.key_available_time[main_key] = time.time() + parsed_delay
                        print(f"[GEMINI-ROTATOR] Key {self.current_index + 1} exhausted. Cooldown set for {parsed_delay:.2f}s.")
                    else:
                        print(f"[GEMINI-ROTATOR] Key {self.current_index + 1} hit a hard quota with NO retry delay. Aborting rotator to trigger model fallback.")
                        break
                elif ("503" in e_str or "UNAVAILABLE" in e_str
                      or "timeout" in e_str.lower() or "timed out" in e_str.lower()
                      or "deadline" in e_str.lower()):
                    # 503 UNAVAILABLE (and request timeouts) are MODEL-wide health problems,
                    # not per-key quotas: every key talks to the same overloaded model, so
                    # rotating keys can't fix it. Try at most one full pass of keys (in case
                    # it's a transient blip), then bail FAST so the caller escalates to the
                    # NEXT model tier instead of grinding against an endpoint that's down.
                    consecutive_503 += 1
                    self.key_available_time[main_key] = time.time() + 3.0
                    print(f"[GEMINI-ROTATOR] Key {self.current_index + 1} hit 503/timeout model-overload spike ({consecutive_503}/{len(self.main_keys)}).")
                    if consecutive_503 >= len(self.main_keys):
                        print(f"[GEMINI-ROTATOR] Model overloaded across all {len(self.main_keys)} keys — escalating to next model tier.")
                        self.current_index = next_index
                        self.save_state()
                        break
                    backoff = random.uniform(0.5, 1.0)  # short blip retry, not the quota delay
                else:
                    # For other errors, no cooldown needed, just rotate
                    consecutive_503 = 0

                self.current_index = next_index
                self.save_state()

                print(f"[GEMINI-ROTATOR] Retrying with next key in {backoff:.2f}s...")
                time.sleep(backoff)

        print("[GEMINI-ROTATOR] Giving up on this model (keys exhausted or model overloaded) — caller will try the next tier.")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Gemini API exhausted all configured keys due to rate limits.")
