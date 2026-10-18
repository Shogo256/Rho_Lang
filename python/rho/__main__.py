# SPDX-License-Identifier: Apache-2.0
"""`python -m rho`: the glyph, the version, and where the compiler is."""
import shutil
import subprocess
import sys

BANNER = r"""
   _ __
  | '_ \      ρ (RHO)
  | |_) |     an array language in the line of APL
  | .__/      one source → a native kernel, a JAX module, a circuit
  |_|
"""


def main() -> int:
    from . import rhoc_command

    try:
        command = rhoc_command()
        version = subprocess.run([*command, "--version"], capture_output=True, text=True, check=True).stdout.split()[-1]
        where = " ".join([shutil.which(command[0]) or command[0], *command[1:]])
    except Exception:  # noqa: BLE001 - the banner still shows without a compiler
        version, where = "(rhoc not found)", "-"
    sys.stdout.write(BANNER.replace("ρ (RHO)", f"ρ (RHO) {version}"))
    print(f"  rhoc: {where}")
    print("  from rho import Kernel; Kernel('(INPUT × 2.0) → =')(INPUT=[1, 2, 3])")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
