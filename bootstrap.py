import os
import sys
import subprocess
import shutil

def log(msg, status="INFO"):
    colors = {
        "INFO": "\033[94m[INFO]\033[0m",
        "SUCCESS": "\033[92m[SUCCESS]\033[0m",
        "WARN": "\033[93m[WARN]\033[0m",
        "ERROR": "\033[91m[ERROR]\033[0m"
    }
    # Fallback if ANSI colors not supported on Windows terminal
    prefix = colors.get(status, f"[{status}]")
    print(f"{prefix} {msg}")

def check_python_version():
    log("Checking Python version...")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        log(f"Python 3.10+ is required. Found Python {major}.{minor}. Please upgrade.", "ERROR")
        sys.exit(1)
    log(f"Python {major}.{minor} detected. (OK)", "SUCCESS")

def setup_virtual_environment():
    venv_dir = os.path.join(os.getcwd(), "env")
    if not os.path.exists(venv_dir):
        log("Virtual environment 'env' not found. Creating a new one...", "INFO")
        try:
            subprocess.run([sys.executable, "-m", "venv", "env"], check=True)
            log("Virtual environment 'env' created successfully.", "SUCCESS")
        except Exception as e:
            log(f"Failed to create virtual environment: {e}", "ERROR")
            sys.exit(1)
    else:
        log("Virtual environment 'env' detected. (OK)", "SUCCESS")

def get_venv_executables():
    # Identify environment python and pip executable paths
    if sys.platform == "win32":
        python_bin = os.path.join("env", "Scripts", "python.exe")
        pip_bin = os.path.join("env", "Scripts", "pip.exe")
    else:
        python_bin = os.path.join("env", "bin", "python")
        pip_bin = os.path.join("env", "bin", "pip")
    
    if not os.path.exists(python_bin) or not os.path.exists(pip_bin):
        log("Virtual environment executables are missing or corrupt. Re-creating environment...", "WARN")
        # Delete corrupt folder and recreate
        try:
            shutil.rmtree("env")
            subprocess.run([sys.executable, "-m", "venv", "env"], check=True)
            log("Virtual environment recreated.", "SUCCESS")
        except Exception as e:
            log(f"Failed to rebuild virtual environment: {e}", "ERROR")
            sys.exit(1)
            
    return python_bin, pip_bin

def install_dependencies(pip_bin):
    requirements_path = "requirements.txt"
    if not os.path.exists(requirements_path):
        log(f"'{requirements_path}' is missing. Skipping package installation.", "WARN")
        return
    
    log("Upgrading pip inside virtual environment...")
    try:
        subprocess.run([pip_bin, "install", "--upgrade", "pip"], check=True, stdout=subprocess.DEVNULL)
        log("Pip upgraded successfully.", "SUCCESS")
    except Exception as e:
        log(f"Failed to upgrade pip: {e}", "WARN")

    log("Installing dependencies from requirements.txt...")
    try:
        subprocess.run([pip_bin, "install", "-r", requirements_path], check=True)
        log("All dependencies installed/verified successfully.", "SUCCESS")
    except Exception as e:
        log(f"Failed to install dependencies: {e}", "ERROR")
        sys.exit(1)

def validate_env_file():
    env_path = ".env"
    required_keys = [
        "GEMINI_API_KEYS",
        "BACKUP_GEMINI_API_KEY",
        "UPSTOX_SANDBOX_ACCESS_TOKEN",
        "UPSTOX_ANALYTICS_ACCESS_TOKEN"
    ]
    
    if not os.path.exists(env_path):
        log(".env configuration file is missing. Generating template...", "WARN")
        template_content = (
            "# --- VANGUARD AI & BROKER KEYS ---\n"
            "GEMINI_API_KEYS=your_gemini_key1_here,your_gemini_key2_here\n"
            "BACKUP_GEMINI_API_KEY=your_backup_gemini_key_here\n"
            "UPSTOX_SANDBOX_ACCESS_TOKEN=your_upstox_sandbox_token_here\n"
            "UPSTOX_ANALYTICS_ACCESS_TOKEN=your_upstox_analytics_token_here\n"
        )
        try:
            with open(env_path, "w") as f:
                f.write(template_content)
            log(f"Template '.env' created. Please edit it and fill in your actual credentials.", "WARN")
        except Exception as e:
            log(f"Failed to write template .env: {e}", "ERROR")
    else:
        log(".env configuration file detected.", "SUCCESS")
        # Load and validate keys
        try:
            with open(env_path, "r") as f:
                lines = f.readlines()
            
            loaded_keys = {}
            for line in lines:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    loaded_keys[k.strip()] = v.strip()
            
            missing_keys = [k for k in required_keys if k not in loaded_keys or not loaded_keys[k]]
            if missing_keys:
                log(f"Missing or empty variables in .env: {', '.join(missing_keys)}", "WARN")
                log("Please open '.env' and fill in those values to ensure correct operations.", "WARN")
            else:
                log("All mandatory environmental variables are configured.", "SUCCESS")
        except Exception as e:
            log(f"Failed to validate .env keys: {e}", "WARN")

def initialize_database(python_bin):
    log("Verifying and initializing database...")
    db_manager_path = os.path.join("scripts", "database_manager.py")
    if not os.path.exists(db_manager_path):
        log("Database manager script 'scripts/database_manager.py' not found! Cannot initialize.", "ERROR")
        return
        
    try:
        # Run init_db using the venv python interpreter so all library modules are present
        cmd = [python_bin, "-c", "from scripts.database_manager import init_db; init_db()"]
        subprocess.run(cmd, check=True)
        log("Database system initialized successfully (SQLite schema & migrations validated).", "SUCCESS")
    except Exception as e:
        log(f"Database initialization failed: {e}", "ERROR")

def main():
    print("\n" + "="*60)
    print("      VANGUARD INTELLIGENCE DASHBOARD BOOTSTRAP PROCESS      ")
    print("="*60 + "\n")
    
    # Enable ANSI escape sequences on Windows 10+ command prompt
    if sys.platform == "win32":
        os.system("")
        
    check_python_version()
    setup_virtual_environment()
    python_bin, pip_bin = get_venv_executables()
    install_dependencies(pip_bin)
    validate_env_file()
    initialize_database(python_bin)
    
    print("\n" + "="*60)
    print("   BOOTSTRAPPING COMPLETE! The system is ready to launch.")
    print("   Use 'bootstrap.bat' or run 'env\\Scripts\\python scripts\\vanguard_dashboard.py'")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
