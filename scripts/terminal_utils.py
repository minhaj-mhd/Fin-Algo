import sys
import os
from datetime import datetime

# ANSI Escape Codes for Premium High-Intensity Neon Colors
class Colors:
    # Standard modifiers
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'
    
    # Custom 256-color high-intensity terminal colors
    GREY = '\033[38;5;243m'         # Soft dim gray for timestamps
    WHITE = '\033[38;5;253m'        # Highly readable off-white for body text
    
    NEON_GREEN = '\033[38;5;82m'    # High-glow green for successes/trades
    NEON_BLUE = '\033[38;5;45m'     # Modern electric blue for info/init
    NEON_CYAN = '\033[38;5;51m'     # Pure cyan for technical stages
    NEON_RED = '\033[38;5;196m'     # Intense red for warnings/errors/vetoes
    NEON_ORANGE = '\033[38;5;208m'  # Safety orange for warnings
    NEON_YELLOW = '\033[38;5;220m'  # Warm gold for key rotation/reset
    NEON_PINK = '\033[38;5;201m'    # Vibrant pink/magenta for exit triggers
    NEON_PURPLE = '\033[38;5;135m'  # Electric purple for scanner operations

# Automatically enable Windows 10+ ANSI color escape codes on module load
if sys.platform == "win32":
    os.system("")
    # Reconfigure stdout/stderr to utf-8 if possible to support unicode characters like Rupee symbol (₹)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def _safe_print(text, **kwargs):
    """
    Safely prints text to the terminal, catching and resolving encoding issues (like CP1252/charmap crashes on Windows).
    Falls back to replacing Rupee symbol (₹) with 'Rs.' and other unencodable characters with ASCII equivalents.
    """
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        # Fallback 1: Replace Rupee symbol with 'Rs.'
        safe_text = text.replace('\u20b9', 'Rs.')
        try:
            print(safe_text, **kwargs)
        except UnicodeEncodeError:
            # Fallback 2: Replace all remaining non-encodable characters
            try:
                stream = kwargs.get('file', sys.stdout)
                encoding = getattr(stream, 'encoding', 'ascii') or 'ascii'
                encoded_bytes = safe_text.encode(encoding, errors='replace')
                decoded_str = encoded_bytes.decode(encoding)
                print(decoded_str, **kwargs)
            except Exception:
                # Final absolute fallback to clean ASCII
                print(safe_text.encode('ascii', errors='replace').decode('ascii'), **kwargs)

# Create centralized file logging
LOG_DIR = os.path.join(os.getcwd(), 'data')
LOG_FILE_PATH = os.path.join(LOG_DIR, 'vanguard_system.log')

def write_to_log_file(msg):
    try:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        
        # Strip ANSI colors just in case
        import re
        clean_msg = re.sub(r'\033\[[0-9;]*m', '', msg)
        
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(clean_msg + '\n')
    except Exception:
        pass

def log(*args, **kwargs):
    """
    Prints a beautifully formatted, colored terminal log with timestamps.
    Colors only the prefix tag, keeping body text distinct and highly readable.
    Writes a plain text copy to data/vanguard_system.log.
    """
    if not args:
        try:
            print(**kwargs)
        except Exception:
            pass
        return

    msg_str = " ".join(map(str, args))
    
    # Premium tag-to-color mapping
    color_map = {
        "[OK]": Colors.NEON_GREEN + Colors.BOLD,
        "[SUCCESS]": Colors.NEON_GREEN + Colors.BOLD,
        "[ERROR]": Colors.NEON_RED + Colors.BOLD,
        "[CRITICAL]": Colors.NEON_RED + Colors.BOLD + Colors.UNDERLINE,
        "[FAIL]": Colors.NEON_RED + Colors.BOLD,
        "[WARN]": Colors.NEON_ORANGE + Colors.BOLD,
        "[INFO]": Colors.NEON_BLUE,
        "[INIT]": Colors.NEON_BLUE + Colors.BOLD,
        "[SCAN]": Colors.NEON_PURPLE + Colors.BOLD,
        "[STAGE 1]": Colors.NEON_CYAN,
        "[STAGE 2]": Colors.NEON_PINK + Colors.BOLD,
        "[VETO]": Colors.NEON_RED + Colors.BOLD,
        "[S1-VETO]": Colors.NEON_RED + Colors.BOLD,
        "[ROTATE]": Colors.NEON_YELLOW + Colors.BOLD,
        "[ROTATE-RESET]": Colors.NEON_YELLOW + Colors.BOLD,
        "[CACHE-HIT]": Colors.GREY,
        "[NETWORK]": Colors.NEON_CYAN + Colors.BOLD,
        "[RESTART]": Colors.NEON_BLUE + Colors.BOLD,
        "[DEBUG]": Colors.GREY,
        "[TRADE]": Colors.NEON_GREEN + Colors.BOLD + Colors.UNDERLINE,
        "[EXIT]": Colors.NEON_PINK + Colors.BOLD,
    }
    
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # 1. Separator lines check (keep clean, no timestamp prefix, themed color)
    if "===" in msg_str or "---" in msg_str:
        _safe_print(f"{Colors.NEON_BLUE}{Colors.BOLD}{msg_str}{Colors.ENDC}", **kwargs)
        write_to_log_file(msg_str)
        return
        
    # 2. Check if the message starts with any of the defined tags
    for tag, color in color_map.items():
        if msg_str.startswith(tag):
            # Extract the remaining message body
            remaining = msg_str[len(tag):]
            
            # Print timestamp, colored tag, and rest of message in soft white
            _safe_print(f"{Colors.GREY}[{current_time}]{Colors.ENDC} {color}{tag}{Colors.ENDC}{Colors.WHITE}{remaining}{Colors.ENDC}", **kwargs)
            write_to_log_file(f"[{current_time}] {tag}{remaining}")
            return

    # 3. Default log format for raw text lines (prefix with soft gray timestamp)
    _safe_print(f"{Colors.GREY}[{current_time}]{Colors.ENDC} {Colors.WHITE}{msg_str}{Colors.ENDC}", **kwargs)
    write_to_log_file(f"[{current_time}] {msg_str}")

def color_text(text, color_code):
    """Wraps text in a specific color code."""
    return f"{color_code}{text}{Colors.ENDC}"

# Test colors if run directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("         VANGUARD PREMIUM TERMINAL LOGS INITIALIZATION       ")
    print("="*60 + "\n")
    
    log("[INIT] Vanguard terminal log system loaded.")
    log("[INFO] Connecting to broker sandbox feed...")
    log("[SUCCESS] Upstox connection established.")
    log("[SCAN] Querying Nifty 200 universe for momentum breakouts.")
    log("[CACHE-HIT] Loaded indicator matrix from disk cache.")
    log("[STAGE 1] Running cross-sectional quantitative filters...")
    log("[OK] Found breakout candidates.")
    log("[STAGE 2] Submitting candidates to Gemini Catalyst Analyst...")
    log("[VETO] VETOED TATASTEEL: Catalyst grounded news confirms regulatory risk.")
    log("[TRADE] Entering LONG position for RELIANCE at \u20b92450.00")
    log("[WARN] Gemini API key 2 returned 429 rate limit.")
    log("[ROTATE] Swapping active API credential to key 3.")
    log("[DEBUG] Current capital: \u20b91,24,000. Available margin: \u20b985,000.")
    log("[EXIT] Exiting position for RELIANCE at \u20b92472.50 | Net: +0.92%")
    
    print("\n" + "="*60)
    print("              TERMINAL COLOR SCHEME TEST COMPLETE            ")
    print("="*60 + "\n")
