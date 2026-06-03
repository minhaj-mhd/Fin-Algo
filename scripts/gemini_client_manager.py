import os
import json
from google import genai

class GeminiRotator:
    def __init__(self, state_file="data/gemini_rotation_state.json"):
        self.state_file = state_file
        self.main_keys = []
        self.backup_key = None
        self.current_index = 0
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
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.current_index = int(data.get("current_index", 0))
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
                json.dump({"current_index": self.current_index}, f)
        except Exception as e:
            print(f"[GEMINI-ROTATOR] Warning: Failed to save rotation state: {e}")

    def execute(self, func, *args, **kwargs):
        """
        Executes a callable 'func(client, *args, **kwargs)' where client is a genai.Client.
        Always uses the current rotated main key first. If it encounters ANY error,
        it uses the BACKUP_GEMINI_API_KEY for that call.
        Regardless of success or failure of the main key, it rotates the pointer to the next main key
        for the NEXT request.
        """
        # Reload keys in case of live environment changes
        self.load_keys()
        
        if not self.main_keys:
            if self.backup_key:
                print("[GEMINI-ROTATOR] No main keys configured. Using backup key directly.")
                client = genai.Client(api_key=self.backup_key)
                return func(client, *args, **kwargs)
            else:
                raise ValueError("No Gemini API keys (main or backup) are configured in the environment.")

        # Identify current main key
        main_key = self.main_keys[self.current_index]
        
        # Prepare the next index
        next_index = (self.current_index + 1) % len(self.main_keys)
        
        try:
            print(f"[GEMINI-ROTATOR] Attempting request using main key {self.current_index + 1} of {len(self.main_keys)}")
            client = genai.Client(api_key=main_key)
            result = func(client, *args, **kwargs)
            
            # Succeeded on main key. Rotate main index for the next request.
            self.current_index = next_index
            self.save_state()
            return result
            
        except Exception as e:
            print(f"[GEMINI-ROTATOR] Main key failed: {e}")
            
            # If main key fails, try backup key at that time
            if self.backup_key:
                print("[GEMINI-ROTATOR] Fallback: Attempting request using backup API key...")
                try:
                    backup_client = genai.Client(api_key=self.backup_key)
                    result = func(backup_client, *args, **kwargs)
                    
                    # Update index for the next request (rotate main key)
                    self.current_index = next_index
                    self.save_state()
                    return result
                except Exception as backup_err:
                    print(f"[GEMINI-ROTATOR] Backup key also failed: {backup_err}")
                    # Still rotate main key for the next request
                    self.current_index = next_index
                    self.save_state()
                    raise backup_err
            else:
                print("[GEMINI-ROTATOR] No backup API key configured. Cannot fall back.")
                # Still rotate main key for the next request
                self.current_index = next_index
                self.save_state()
                raise e
