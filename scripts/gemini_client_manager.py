import os
import time
import json
from google import genai

class GeminiRotator:
    def __init__(self, state_file="data/gemini_rotation_state.json"):
        self.state_file = state_file
        self.main_keys = []
        self.backup_key = None
        self.current_index = 0
        self.stats = {}
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
        
        import re

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
                client = genai.Client(api_key=main_key)
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
                
                # Parse retry delay if it's a 429 RESOURCE_EXHAUSTED
                if "429" in e_str and "RESOURCE_EXHAUSTED" in e_str:
                    match = re.search(r'Please retry in ([0-9.]+)s', e_str)
                    if match:
                        parsed_delay = float(match.group(1))
                        self.key_available_time[main_key] = time.time() + parsed_delay
                        print(f"[GEMINI-ROTATOR] Key {self.current_index + 1} exhausted. Cooldown set for {parsed_delay:.2f}s.")
                    else:
                        print(f"[GEMINI-ROTATOR] Key {self.current_index + 1} hit a hard quota with NO retry delay. Aborting rotator to trigger model fallback.")
                        break
                elif "503" in e_str or "UNAVAILABLE" in e_str:
                    # Temporary spike, let's just rotate with a small cooldown
                    self.key_available_time[main_key] = time.time() + 3.0
                    print(f"[GEMINI-ROTATOR] Key {self.current_index + 1} hit 503 spike. Cooldown set for 3.00s.")
                else:
                    # For other errors, no cooldown needed, just rotate
                    pass
                
                self.current_index = next_index
                self.save_state()
                
                # Apply a 3-5 second random delay between failed requests before trying the next key
                import random
                delay = random.uniform(3.0, 5.0)
                print(f"[GEMINI-ROTATOR] Random delay of {delay:.2f}s before trying the next key...")
                time.sleep(delay)

        print("[GEMINI-ROTATOR] All attempts exhausted or cooldown too long. API permanently unavailable.")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Gemini API exhausted all configured keys due to rate limits.")
