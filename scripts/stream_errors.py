import os
import sys
import time

# Add project root to path so we can import local modules
sys.path.append(os.getcwd())

from scripts.terminal_utils import Colors as C, _safe_print

LOG_FILE = os.path.join(os.getcwd(), 'data', 'vanguard_system.log')
FILTER_TAGS = ["[ERROR]", "[WARN]", "[CRITICAL]", "[FAIL]", "[VETO]", "[S1-VETO]"]

def main():
    if not os.path.exists(LOG_FILE):
        # Ensure directory and empty file exist
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"--- VANGUARD SYSTEM LOG MONITOR STARTED {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            
    # Clear screen only if connected to a real terminal/TTY
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{C.NEON_BLUE}{C.BOLD}======================================================================{C.ENDC}")
    print(f"  {C.NEON_RED}{C.BOLD}VANGUARD SYSTEM — REAL-TIME ERROR & WARNING MONITOR{C.ENDC}")
    print(f"{C.NEON_BLUE}{C.BOLD}======================================================================{C.ENDC}")
    print(f"  {C.GREY}Streaming live log output from: {LOG_FILE}{C.ENDC}")
    print(f"  {C.GREY}Filters active: {', '.join([C.NEON_YELLOW + t + C.GREY for t in FILTER_TAGS])}{C.ENDC}")
    print(f"{C.NEON_BLUE}======================================================================{C.ENDC}\n")

    # Open log file and seek to the end
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        f.seek(0, os.SEEK_END)
        
        while True:
            line = f.readline()
            if not line:
                # Essential for Windows: reset EOF status and flush buffers to see new appends
                f.seek(f.tell())
                time.sleep(0.1) # Non-blocking sleep
                continue
                
            line = line.strip()
            if not line:
                continue
                
            # Match any of our warning/error filter tags
            matched_tag = None
            for tag in FILTER_TAGS:
                if tag in line:
                    matched_tag = tag
                    break
                    
            if matched_tag:
                # Apply appropriate color based on tag
                color = C.NEON_ORANGE if matched_tag == "[WARN]" else C.NEON_RED
                if matched_tag == "[CRITICAL]":
                    color += C.UNDERLINE
                
                # Format to keep the timestamp grey and colorize only the matched tag
                parts = line.split(matched_tag, 1)
                if len(parts) == 2:
                    timestamp_part = parts[0].strip()
                    body_part = parts[1]
                    _safe_print(f"{C.GREY}{timestamp_part}{C.ENDC} {color}{C.BOLD}{matched_tag}{C.ENDC}{C.WHITE}{body_part}{C.ENDC}")
                else:
                    _safe_print(f"{color}{C.BOLD}{line}{C.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.NEON_BLUE}[INFO] Log monitor stopped by user.{C.ENDC}")
