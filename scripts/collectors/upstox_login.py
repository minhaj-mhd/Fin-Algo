"""
Upstox OAuth login helper — mints a FULL-ACCESS daily access token.

Why: the analytics/extended token is READ-ONLY and the expired-instruments OI
APIs reject it with UDAPI100067. The full-access token comes from the interactive
login flow (browser login is yours; this script does everything around it).

Prereqs in .env (gitignored — never paste these in chat):
    UPSTOX_API_KEY=...
    UPSTOX_API_SECRET=...
    UPSTOX_REDIRECT_URI=...      # must match your registered Upstox app

Usage (run in YOUR terminal — step 2 needs your browser):
    1) python scripts/collectors/upstox_login.py
         -> prints the authorize URL. Open it, log in, approve.
            Upstox redirects to  {REDIRECT_URI}?code=XXXX
    2) python scripts/collectors/upstox_login.py "<paste full redirect URL or just the code>"
         -> exchanges the code, decodes isPlus/exp, writes
            UPSTOX_FULL_ACCESS_TOKEN=... back into .env

Writes only .env (UTF-8, in place). Touches nothing under data/, models/, or the vault.
"""
import os
import sys
import json
import base64
import datetime as dt
from urllib.parse import urlparse, parse_qs

import requests
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
AUTH_DIALOG = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
TARGET_VAR = "UPSTOX_FULL_ACCESS_TOKEN"


def cfg():
    load_dotenv()
    key = os.getenv("UPSTOX_API_KEY")
    secret = os.getenv("UPSTOX_API_SECRET")
    redirect = os.getenv("UPSTOX_REDIRECT_URI")
    missing = [n for n, v in [("UPSTOX_API_KEY", key), ("UPSTOX_API_SECRET", secret),
                              ("UPSTOX_REDIRECT_URI", redirect)] if not v]
    if missing:
        sys.exit(f"FAIL: set these in .env first: {', '.join(missing)}")
    return key, secret, redirect


def decode_claims(jwt):
    """Decode a JWT payload (no signature verification) for isPlus/exp/sub."""
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)            # pad base64
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception as e:
        return {"_decode_error": str(e)}


def extract_code(arg):
    """Accept either a bare code or a full redirect URL containing ?code=..."""
    if "code=" in arg or arg.startswith("http"):
        qs = parse_qs(urlparse(arg).query)
        if qs.get("code"):
            return qs["code"][0]
    return arg.strip()


def upsert_env(var, value):
    """Replace or append `var=value` in .env, preserving everything else (UTF-8)."""
    path = os.path.abspath(ENV_PATH)
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    out, found = [], False
    for ln in lines:
        if ln.startswith(var + "="):
            out.append(f"{var}={value}"); found = True
        else:
            out.append(ln)
    if not found:
        out.append(f"{var}={value}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"  wrote {var} to {path}")


def main():
    key, secret, redirect = cfg()

    # No code yet -> print the authorize URL and stop.
    if len(sys.argv) < 2:
        url = (f"{AUTH_DIALOG}?response_type=code&client_id={key}"
               f"&redirect_uri={redirect}")
        print("STEP 1 — open this URL, log in, approve:\n")
        print("  " + url + "\n")
        print("Then re-run with the code:")
        print('  python scripts/collectors/upstox_login.py "<redirect URL or code>"')
        return

    code = extract_code(sys.argv[1])
    print(f"Exchanging code (len={len(code)}) for access token...")
    r = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"},
        data={"code": code, "client_id": key, "client_secret": secret,
              "redirect_uri": redirect, "grant_type": "authorization_code"},
        timeout=30,
    )
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    token = body.get("access_token")
    if not token:
        sys.exit(f"FAIL http={r.status_code}: {json.dumps(body)[:400]}")

    claims = decode_claims(token)
    is_plus = claims.get("isPlusPlan", "?")     # actual claim name is isPlusPlan
    is_ext = claims.get("isExtended", "?")
    exp = claims.get("exp")
    exp_str = dt.datetime.fromtimestamp(exp).isoformat() if isinstance(exp, (int, float)) else "?"
    print("  token OK.")
    print(f"  isPlusPlan = {is_plus}   (must be True for /expired-instruments/*)")
    print(f"  isExtended = {is_ext}    (must be False/None — extended tokens are read-only)")
    print(f"  expires = {exp_str}   sub = {claims.get('sub', '?')}")
    upsert_env(TARGET_VAR, token)
    if is_plus is not True or is_ext is True:
        print("  WARNING: need isPlusPlan=True AND isExtended!=True, else OI stays blocked.")
    print("\nNext: python scripts/collectors/probe_upstox_oi.py  (point it at the new token)")


if __name__ == "__main__":
    main()
