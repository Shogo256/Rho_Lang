#!/usr/bin/env python3
"""Randomised soundness check for the ! constraint solver.

Ordinary tests check that the solver answers known cases correctly. This checks
something the test suite cannot: that the solver never claims a proof the
hardware disproves. For each generated program it asks the solver, then runs the
compiled kernel over many inputs and compares.

Two failures are reported, and both have actually happened:

  unsound proof   the solver said `proved` and a real run violated it.
                  Found a case where the solver reasoned over ℝ while the
                  kernel computed in binary64: (x*x)/x > x is impossible over
                  the reals but not on the machine.

  contradiction   one backend proved what the other rejected. At most one can be
                  right, and a wrong rejection blocks a valid program. Found a
                  case where the two reads of the same shifted cell were given
                  independent boundary flags.

Usage:
    scripts/fuzz_solver.py [seed] [rounds]

Set RHOC_INTERVAL and RHOC_Z3 to compiled drivers to compare both backends;
with only one available it still checks that backend for unsound proofs.
"""

import ctypes
import os
import random
import subprocess
import sys

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
CELLS = 24

BACKENDS = {}
for name, env in (("interval", "RHOC_INTERVAL"), ("z3", "RHOC_Z3")):
    path = os.environ.get(env)
    if path and os.path.exists(path):
        BACKENDS[name] = os.path.abspath(path)
if not BACKENDS:
    default = os.path.join("target", "release", "rhoc")
    if not os.path.exists(default):
        sys.exit("no rhoc found; build it or set RHOC_INTERVAL / RHOC_Z3")
    BACKENDS["interval"] = os.path.abspath(default)

random.seed(SEED)


def expr(depth, spaces):
    if depth <= 0 or random.random() < 0.25:
        return random.choice([
            random.choice(spaces),
            f"{random.choice([0.0, 1.0, 2.0, 0.5, -1.0, 3.0]):.1f}",
        ])
    r = random.random()
    if r < 0.22:
        return f"({random.choice(['▷', '▽'])}{random.choice(['', '0'])}{random.choice(spaces)})"
    if r < 0.34:
        return f"({expr(depth - 1, spaces)} ^ {random.choice([2.0, 3.0, 4.0]):.1f})"
    op = random.choice(["+", "-", "×", "/", ">", "<"])
    return f"({expr(depth - 1, spaces)} {op} {expr(depth - 1, spaces)})"


def generate():
    flows, spaces = [], ["INPUT"]
    for k in range(random.randint(1, 3)):
        name = f"T{k}"
        flows.append(f"    {expr(3, tuple(spaces))} → {name}")
        spaces.append(name)
    flows.append(f"    {expr(3, tuple(spaces))} → OUTPUT")
    cmp_ = random.choice([">=", ">", "<=", "<"])
    flows.append(f"    ! (OUTPUT {cmp_} 0)")
    flows.append("    OUTPUT → =")
    return "{\n    INPUT:◯ □ 6 4\n" + "\n".join(flows) + "\n}\n", cmp_


def holds(v, cmp_):
    return {">=": v >= 0, ">": v > 0, "<=": v <= 0}.get(cmp_, v < 0)


def run(so, data):
    lib = ctypes.CDLL(so)
    fn = lib.rho_kernel_exec_with_args
    fn.argtypes = [ctypes.POINTER(ctypes.c_double)] * 2
    fn.restype = None
    a = (ctypes.c_double * CELLS)(*data)
    o = (ctypes.c_double * CELLS)()
    fn(a, o)
    return list(o)


def compile_with(driver, rho, so):
    r = subprocess.run([driver, rho, "-o", so], capture_output=True, text=True)
    if r.returncode == 0:
        return "proved" if "[proved]   (OUTPUT" in r.stdout else "unproven"
    if "Logic Failure" not in r.stderr:
        return "error"
    # Only a constraint rejection is comparable across backends; a division
    # rejection answers a different question.
    return "violated" if "OUTPUT " in r.stderr else "division-reject"


def main():
    unsound, contradictions, analysed = [], [], 0

    for i in range(ROUNDS):
        source, cmp_ = generate()
        rho = f"_fuzz{i}.rho"
        with open(rho, "w", encoding="utf-8") as f:
            f.write(source)

        verdicts, artefact = {}, None
        for name, driver in BACKENDS.items():
            so = os.path.abspath(f"_fuzz{i}_{name}.so")
            verdicts[name] = compile_with(driver, rho, so)
            if os.path.exists(so):
                artefact = so
        if "error" in verdicts.values():
            os.remove(rho)
            continue
        analysed += 1

        violation = None
        if artefact:
            for t in range(20):
                if t == 0:
                    data = [0.0] * CELLS
                elif t == 1:
                    data = [float(k) for k in range(CELLS)]
                elif t == 2:
                    data = [-float(k) for k in range(CELLS)]
                else:
                    data = [random.uniform(-50, 50) for _ in range(CELLS)]
                for v in run(artefact, data):
                    if v == v and not holds(v, cmp_):
                        violation = v
                        break
                if violation is not None:
                    break

        for name, verdict in verdicts.items():
            if verdict == "proved" and violation is not None:
                unsound.append((name, source, violation))

        settled = {v for v in verdicts.values() if v != "division-reject"}
        if "proved" in settled and "violated" in settled:
            contradictions.append((dict(verdicts), source))

        for path in [rho] + [f"_fuzz{i}_{n}.so" for n in BACKENDS]:
            if os.path.exists(path):
                os.remove(path)

    print(f"seed {SEED}: generated {ROUNDS}, analysed {analysed}")
    print(f"  backends: {', '.join(sorted(BACKENDS))}")
    print(f"  unsound proofs (proved, then disproved by a run): {len(unsound)}")
    for name, source, value in unsound[:3]:
        print(f"    [{name}] produced {value}\n{source}")
    print(f"  contradictions (one proved, another rejected):    {len(contradictions)}")
    for verdicts, source in contradictions[:3]:
        print(f"    {verdicts}\n{source}")

    return 1 if (unsound or contradictions) else 0


if __name__ == "__main__":
    sys.exit(main())
