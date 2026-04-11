#!/usr/bin/env python3
"""Generate machine-bound activation keys for SIMPLIAI.

Use this in two ways:
1) Generate a master secret (store as SIMPLIAI_PASSKEY on target machines).
2) Generate a machine-bound activation key from a system code + master secret.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import os
import secrets
import uuid
from pathlib import Path


def make_key(num_bytes: int = 48) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(num_bytes)).decode("ascii").rstrip("=")


def make_activation_key(system_code: str, master_secret: str) -> str:
    sc = str(system_code or "").strip().upper()
    if not sc:
        raise ValueError("System code is required")
    sec = str(master_secret or "").strip()
    if not sec:
        raise ValueError("Master secret is required")
    digest = hmac.new(sec.encode("utf-8"), sc.encode("utf-8"), hashlib.sha256).hexdigest()[:24].upper()
    return f"{sc}-{digest}"


def local_system_code() -> str:
    machine_hint = f"{uuid.getnode()}:{os.environ.get('COMPUTERNAME', '')}"
    return hashlib.sha256(machine_hint.encode("utf-8")).hexdigest()[:12].upper()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SIMPLIAI secrets and machine-bound activation keys")
    parser.add_argument(
        "-o",
        "--output",
        default="activation_passkey.txt",
        help="Output file path (default: activation_passkey.txt)",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=48,
        help="Random bytes before encoding (default: 48)",
    )
    parser.add_argument(
        "--system-code",
        default="",
        help="Target machine system code (from app activation status)",
    )
    parser.add_argument(
        "--master-secret",
        default="",
        help="Master secret; if omitted, reads SIMPLIAI_PASSKEY env var",
    )
    parser.add_argument(
        "--print-system-code",
        action="store_true",
        help="Print this machine's system code and exit",
    )
    args = parser.parse_args()

    if args.bytes < 16:
        raise SystemExit("--bytes must be >= 16")

    if args.print_system_code:
        print(local_system_code())
        return 0

    # Default interactive mode: ask for system code and generate shareable key.
    if not args.system_code and not args.master_secret:
        print("Interactive activation key generator")
        print("Paste target machine system code from app/generate_passkey.py --print-system-code")
        entered_code = input("System code: ").strip().upper()
        if not entered_code:
            raise SystemExit("System code is required")

        master_secret = os.environ.get("SIMPLIAI_PASSKEY", "").strip()
        if not master_secret:
            master_secret = getpass.getpass("Master secret (hidden): ").strip()
        if not master_secret:
            raise SystemExit("Master secret is required")

        key = make_activation_key(entered_code, master_secret)
        out_path = Path(args.output).resolve()
        out_path.write_text(key + "\n", encoding="utf-8")
        print(f"\nMachine-bound activation key saved to: {out_path}")
        print(f"Activation key to share: {key}")
        return 0

    # Mode A: machine-bound key generation
    if args.system_code:
        master_secret = args.master_secret or os.environ.get("SIMPLIAI_PASSKEY", "")
        if not master_secret:
            raise SystemExit("Provide --master-secret or set SIMPLIAI_PASSKEY in environment")

        key = make_activation_key(args.system_code, master_secret)
        out_path = Path(args.output).resolve()
        out_path.write_text(key + "\n", encoding="utf-8")
        print(f"Machine-bound activation key saved to: {out_path}")
        print(f"System code: {args.system_code.strip().upper()}")
        print(f"Activation key: {key}")
        return 0

    # Mode B: master secret generation
    key = make_key(args.bytes)
    out_path = Path(args.output).resolve()
    out_path.write_text(key + "\n", encoding="utf-8")

    print(f"Master secret generated and saved to: {out_path}")
    print("\nPowerShell (persist for current user):")
    print(f"[Environment]::SetEnvironmentVariable('SIMPLIAI_PASSKEY', '{key}', 'User')")
    print("\nPowerShell (current terminal only):")
    print(f"$env:SIMPLIAI_PASSKEY = '{key}'")
    print("\nThen generate a machine key:")
    print("python generate_passkey.py --system-code <SYSTEM_CODE> --master-secret <MASTER_SECRET>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
