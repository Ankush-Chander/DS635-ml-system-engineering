#!/usr/bin/env python3
"""
grade_submissions.py -- automated marking for the Lecture 5/6 GPU submission lab.

    python grade_submissions.py <submissions_dir> [-o report_dir] [--csv-only]

WHAT THIS CAN AND CANNOT MARK
-----------------------------
Every number in the lab is machine-dependent by design: a Colab T4, an RX 6700M
and an RTX 3060 disagree on every cell. So there is no answer key.

What IS gradable is that the lab validates a physical model, and that model makes
*ordinal* predictions which hold on any GPU:

    submission << execution        the two honest timers agree
    cache rescues small workloads  draining a queue costs
    streams overlap                launch cost is roughly constant

This script marks those relationships between a student's OWN numbers (40 marks)
and extracts their prose for rubric marking (60 marks). It never compares a
student's magnitudes against a reference.

TRAPS DELIBERATELY NOT CHECKED
------------------------------
- **Pinned vs pageable direction.** Measured on the course machine: 1.48x at 1 MB
  but ~1.00x at >=64 MB. A student reporting "no speedup at large sizes" is
  CORRECT. Asserting pinned > pageable would fail them for being right.
- **CUDA Graph GPU-side improvement.** Graph replay was 8.2x SLOWER on the course
  machine's ROCm/RDNA2 stack. Only the CPU submission column is checked, and only
  when capture actually succeeded.
- **overlap >= max(copy, compute).** Looks obviously true; the course machine
  measured overlap 23.8 ms against a max of 25.0 ms, because the three timings are
  separate noisy runs. A8 tests `overlap < 0.95 * serial` instead, and is skipped
  (marks awarded, flagged as inconclusive) when copy and compute are too unbalanced
  on that student's GPU for overlap to be observable at all.
- **Anything the backend cannot do.** Apple's Metal backend exposes no command
  streams, no graph capture and no pinned host memory. Those blocks arrive marked
  `{"skipped": ...}`, count as recorded for completeness, and their dependent checks
  award full marks with a flag. A student is never penalised for their hardware.
- **Route A of the peak task on Apple silicon.** Metal reports no core count and no
  clock, so a derived peak is impossible there; A11 credits its *absence* instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

AUTO_TOTAL = 40
RUBRIC_TOTAL = 60

REQUIRED_BLOCKS = [
    "peak", "submission", "launch", "sweep", "fence", "transfer", "overlap", "graph",
]


def was_skipped(R, name):
    """True if the backend could not run this experiment (Apple MPS, old driver)."""
    b = R.get(name)
    return isinstance(b, dict) and bool(b.get("skipped"))


@dataclass
class Check:
    cid: str
    label: str
    marks: float
    earned: float = 0.0
    ok: bool | None = None
    detail: str = ""


@dataclass
class Submission:
    path: Path
    roll: str
    name: str
    data: dict
    nb_path: Path | None = None
    checks: list[Check] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    prose: dict = field(default_factory=dict)

    @property
    def auto_score(self) -> float:
        return round(sum(c.earned for c in self.checks), 2)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def g(d, *path, default=None):
    """Nested get that tolerates missing branches."""
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def num(x):
    """Coerce to float, or None if not a usable number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def close(a, b, tol=0.25):
    a, b = num(a), num(b)
    if a is None or b is None or b == 0:
        return False
    return abs(a - b) / abs(b) <= tol


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def run_checks(s: Submission) -> None:
    d, R = s.data, s.data.get("results", {})
    env, ans = d.get("env", {}), d.get("answers", {})
    add = s.checks.append

    # --- A1 provenance -----------------------------------------------------
    c = Check("A1", "Provenance: device, backend, declared peak, identity", 4)
    have = [bool(env.get("device")), bool(env.get("torch")),
            (num(env.get("peak_tflops_declared")) or 0) > 0, bool(s.roll)]
    c.earned = round(c.marks * sum(have) / len(have), 2)
    c.ok = all(have)
    c.detail = (f"device={env.get('device')!r} torch={env.get('torch')!r} "
                f"peak={env.get('peak_tflops_declared')} roll={s.roll!r}")
    add(c)

    # --- A2 completeness ---------------------------------------------------
    c = Check("A2", "All experiments recorded (skipped-by-backend counts)", 3)
    present = [b for b in REQUIRED_BLOCKS if b in R]
    c.earned = round(c.marks * len(present) / len(REQUIRED_BLOCKS), 2)
    c.ok = len(present) == len(REQUIRED_BLOCKS)
    missing = sorted(set(REQUIRED_BLOCKS) - set(present))
    c.detail = "all present" if c.ok else f"missing: {missing}"
    add(c)

    p1 = R.get("submission", {})
    t_nosync, t_sync, t_event = (num(p1.get(k)) for k in
                                 ("t_nosync_ms", "t_sync_ms", "t_event_ms"))

    # --- A3 submission is not execution ------------------------------------
    c = Check("A3", "Unsynchronised timer >=10x faster than synchronised", 4)
    if t_nosync and t_sync and t_nosync > 0:
        ratio = t_sync / t_nosync
        c.ok = ratio >= 10
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"t_sync/t_nosync = {ratio:.1f}x (need >=10)"
    else:
        c.detail = "part1 timings missing"
    add(c)

    # --- A4 the impossible number ------------------------------------------
    c = Check("A4", "Implied no-sync throughput exceeds advertised peak", 4)
    peak = num(env.get("peak_tflops_declared"))
    implied = num(g(p1, "implied_tflops", "nosync"))
    if implied is None and t_nosync and num(p1.get("flop")):
        implied = num(p1["flop"]) / (t_nosync * 1e-3) / 1e12
    if implied and peak:
        c.ok = implied > peak
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"implied {implied:,.0f} TF/s vs peak {peak} TF/s ({implied/peak:.0f}x)"
    else:
        c.detail = "cannot compute implied throughput"
    add(c)

    # --- A5 independent timers agree ---------------------------------------
    c = Check("A5", "synchronize() and CUDA events agree within 15%", 4)
    if t_sync and t_event:
        rel = abs(t_sync - t_event) / t_sync
        c.ok = rel <= 0.15
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"|sync-event|/sync = {rel*100:.1f}% (need <=15%)"
    else:
        c.detail = "missing sync or event timing"
    add(c)

    # --- A6 launch-bound signature -----------------------------------------
    c = Check("A6", "Launch-bound signature in the size sweep", 4)
    sweep, submit_us = R.get("sweep", {}), num(g(R, "launch", "submit_us"))
    per_op = [num(x) for x in (sweep.get("per_op_us") or [])]
    per_op = [x for x in per_op if x is not None]
    if len(per_op) >= 3 and submit_us:
        rises = per_op[-1] > per_op[0] * 2
        flat_matches_submit = 0.3 <= (per_op[0] / submit_us) <= 3.0
        earned = c.marks * (0.6 * rises + 0.4 * flat_matches_submit)
        c.earned, c.ok = round(earned, 2), (rises and flat_matches_submit)
        c.detail = (f"tail/head = {per_op[-1]/per_op[0]:.1f}x (need >2); "
                    f"head/submit = {per_op[0]/submit_us:.2f} (need 0.3-3.0)")
    else:
        c.detail = "sweep or submission cost missing"
    add(c)

    # --- A7 fence cost ------------------------------------------------------
    c = Check("A7", "Syncing every op costs >=1.5x syncing once", 4)
    fence = R.get("fence", {})
    iv = [num(x) for x in (fence.get("intervals") or [])]
    tot = [num(x) for x in (fence.get("totals_ms") or [])]
    if iv and tot and len(iv) == len(tot) and min(tot) > 0:
        drained = tot[iv.index(min(iv))]
        batched = tot[iv.index(max(iv))]
        ratio = drained / batched
        c.ok = ratio >= 1.5
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"drained/batched = {ratio:.2f}x (need >=1.5)"
    else:
        c.detail = "fence sweep missing or malformed"
    add(c)

    # --- A8 overlap ---------------------------------------------------------
    # `overlap >= max(copy, compute)` is NOT asserted -- the course machine
    # violated it through run-to-run variance. See the module docstring.
    c = Check("A8", "Two streams beat one stream", 4)
    ov = R.get("overlap", {})
    if was_skipped(R, "overlap"):
        c.ok, c.earned = True, c.marks
        c.detail = f"skipped by backend ({ov['skipped']}) -- marks awarded"
        s.flags.append("overlap skipped: backend has no stream API")
    else:
        cp, cm, ser, ovl = (num(ov.get(k)) for k in
                            ("copy_ms", "compute_ms", "serial_ms", "overlap_ms"))
        if None in (cp, cm, ser, ovl) or ovl <= 0 or max(cp, cm) <= 0:
            c.detail = "overlap block missing or malformed"
        else:
            # Gate: overlap is only observable when the two halves are comparable.
            # Where compute dwarfs the copy (or vice versa) serial ~= overlap
            # legitimately, and failing the student for their hardware is wrong.
            balance = min(cp, cm) / max(cp, cm)
            if balance < 0.2:
                c.ok, c.earned = True, c.marks
                c.detail = (f"inconclusive on this device (copy {cp:.2f} vs compute "
                            f"{cm:.2f}, balance {balance:.2f} < 0.20) -- marks awarded")
                s.flags.append("overlap check inconclusive: copy and compute not comparable")
            else:
                c.ok = ovl < 0.95 * ser
                c.earned = c.marks if c.ok else 0.0
                c.detail = (f"overlap {ovl:.2f} vs serial {ser:.2f} "
                            f"({ser/ovl:.2f}x, need >1.05x); sum was {cp+cm:.2f}")
    add(c)

    # --- A9 launch cost plausible ------------------------------------------
    c = Check("A9", "Per-launch submission cost in 1-20 us", 2)
    if submit_us:
        c.ok = 1.0 <= submit_us <= 20.0
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"submit = {submit_us:.2f} us"
    else:
        c.detail = "submission cost missing"
    add(c)

    # --- A10 answers consistent with own data ------------------------------
    # Verifies the student actually read their own numbers.
    c = Check("A10", "Short answers consistent with recorded measurements", 3)
    sub = []
    pk = R.get("peak", {})
    if num(pk.get("measured_tflops")) and num(pk.get("vendor_tflops")):
        sub.append(("peak_efficiency",
                    close(ans.get("peak_efficiency"),
                          num(pk["measured_tflops"]) / num(pk["vendor_tflops"]), 0.30)))
    if peak and implied:
        sub.append(("nosync_over_peak", close(ans.get("nosync_over_peak"), implied / peak, 0.30)))
    if iv and tot and min(tot) > 0:
        sub.append(("fence_penalty_ratio",
                    close(ans.get("fence_penalty_ratio"),
                          tot[iv.index(min(iv))] / tot[iv.index(max(iv))], 0.30)))
    ovb = R.get("overlap", {})
    ocp, ocm, oovl = (num(ovb.get(k)) for k in ("copy_ms", "compute_ms", "overlap_ms"))
    if None not in (ocp, ocm, oovl) and oovl > 0:
        sub.append(("overlap_efficiency",
                    close(ans.get("overlap_efficiency"), (ocp + ocm) / oovl, 0.30)))
    gr = R.get("graph", {})
    if gr.get("captured") and num(gr.get("graph_cpu_ms")):
        sub.append(("graph_cpu_speedup",
                    close(ans.get("graph_cpu_speedup"),
                          num(gr["eager_cpu_ms"]) / num(gr["graph_cpu_ms"]), 0.30)))
    if sub:
        good = sum(1 for _, okk in sub if okk)
        c.earned = round(c.marks * good / len(sub), 2)
        c.ok = good == len(sub)
        c.detail = ", ".join(f"{k}:{'ok' if okk else 'MISMATCH'}" for k, okk in sub)
    else:
        c.detail = "no derivable answers to cross-check"
    add(c)

    # --- A11 the peak task ---------------------------------------------------
    # Route A is impossible on Apple silicon (Metal exposes no core count or
    # clock), so it is credited when absent rather than required.
    c = Check("A11", "Peak established on every available route; measured <= vendor", 4)
    pk = R.get("peak", {})
    vend, meas, deriv = (num(pk.get(k)) for k in
                         ("vendor_tflops", "measured_tflops", "derived_tflops"))
    apple = str(env.get("vendor", "")) == "mps"
    if vend and meas:
        parts = []
        parts.append(("vendor cited", bool(str(pk.get("vendor_source") or "").strip())))
        parts.append(("measured <= vendor", meas <= vend * 1.05))
        parts.append(("efficiency plausible", 0.15 <= meas / vend <= 1.05))
        if apple:
            parts.append(("route A n/a on Metal", deriv is None))
        else:
            parts.append(("route A derived", deriv is not None and deriv > 0))
        good = sum(1 for _, okk in parts)
        hit = sum(1 for _, okk in parts if okk)
        c.earned = round(c.marks * hit / good, 2)
        c.ok = hit == good
        c.detail = (f"vendor {vend} / measured {meas:.2f} "
                    f"({meas/vend*100:.0f}%); derived {deriv}; "
                    + ", ".join(f"{k}:{'ok' if okk else 'FAIL'}" for k, okk in parts))
    else:
        c.detail = "peak block missing vendor or measured value"
    add(c)

    # --- non-marking flags --------------------------------------------------
    if env.get("safe_mode"):
        s.flags.append("SAFE_MODE on (reduced sizes) — expected off mains power")
    if env.get("ac_power") is False:
        s.flags.append("ran on BATTERY — reduced workload, reboot risk")
    if was_skipped(R, "graph"):
        s.flags.append("graphs skipped: no capture API on this backend")
    elif not gr.get("captured", True):
        s.flags.append("graph capture failed (not penalised)")
    if was_skipped(R, "transfer"):
        s.flags.append("pinned-memory comparison skipped: unified memory / no pinning")
    unfilled = [k for k, v in ans.items() if v is None]
    if unfilled:
        s.flags.append(f"{len(unfilled)} short answers left as None: {unfilled}")


# --------------------------------------------------------------------------
# prose extraction from the notebook
# --------------------------------------------------------------------------

PLACEHOLDERS = ("*your answer:*", "*your prediction:*", "your answer here",
                "| | your machine |", "📝")


def extract_prose(nb_path: Path) -> dict:
    """Pull tagged answer/prediction cells out of the executed notebook."""
    try:
        nb = json.loads(nb_path.read_text())
    except Exception as e:  # noqa: BLE001
        return {"_error": f"unreadable notebook: {e}"}
    out = {}
    for cell in nb.get("cells", []):
        tags = cell.get("metadata", {}).get("tags") or []
        if not ({"answer", "prediction"} & set(tags)):
            continue
        task = next((t for t in tags if t.startswith("task-")), None)
        kind = "prediction" if "prediction" in tags else "answer"
        key = f"{task or 'untagged'}:{kind}"
        body = "".join(cell.get("source", []))
        # strip the prompt text so only the student's own writing is measured
        lines = [l for l in body.split("\n")
                 if l.strip() and not any(ph in l.lower() for ph in PLACEHOLDERS)]
        written = "\n".join(lines)
        out[key] = {"chars": len(written.strip()), "text": body.strip()}
    return out


def prose_completeness(prose: dict) -> tuple[int, int]:
    """(filled, total) — a cell counts as filled if it grew beyond its template."""
    cells = {k: v for k, v in prose.items() if not k.startswith("_")}
    if not cells:
        return 0, 0
    # Template-only cells sit well under this; a real answer clears it easily.
    filled = sum(1 for v in cells.values() if v["chars"] > 200)
    return filled, len(cells)


# --------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------

def float_fingerprint(sub: Submission) -> tuple:
    """The measured floats, in a stable order. Two students matching is ~impossible."""
    vals = []
    R = sub.data.get("results", {})
    for block in sorted(R):
        b = R[block]
        if not isinstance(b, dict):
            continue
        for k in sorted(b):
            v = b[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(round(float(v), 6))
            elif isinstance(v, list):
                vals += [round(float(x), 6) for x in v
                         if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return tuple(vals)


def find_duplicates(subs: list[Submission]) -> list[tuple[str, str, int]]:
    """Exact shared measurement vectors — copied outputs."""
    groups = defaultdict(list)
    for s in subs:
        fp = float_fingerprint(s)
        if len(fp) >= 8:
            groups[fp].append(s.roll)
    dupes = []
    for fp, rolls in groups.items():
        if len(rolls) > 1:
            for i in range(len(rolls)):
                for j in range(i + 1, len(rolls)):
                    dupes.append((rolls[i], rolls[j], len(fp)))
    return dupes


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def scorecard(s: Submission) -> str:
    L = [f"# Lab 5/6 scorecard — {s.roll}" + (f" ({s.name})" if s.name else ""), ""]
    env = s.data.get("env", {})
    L += [f"- **Device:** {env.get('device','?')}",
          f"- **Backend:** torch {env.get('torch','?')} / {env.get('backend','?')}",
          f"- **Declared peak:** {env.get('peak_tflops_declared','?')} TFLOP/s",
          f"- **Run:** {env.get('started_utc','?')} → {env.get('exported_utc','?')}",
          f"- **Notebook:** {s.nb_path.name if s.nb_path else '**MISSING**'}", ""]
    L += [f"## Automatic checks — {s.auto_score} / {AUTO_TOTAL}", "",
          "| ID | Check | Marks | Result | Detail |", "|---|---|---:|---|---|"]
    for c in s.checks:
        mark = "PASS" if c.ok else ("PARTIAL" if 0 < c.earned < c.marks else "FAIL")
        L.append(f"| {c.cid} | {c.label} | {c.earned:g}/{c.marks:g} | {mark} | {c.detail} |")
    L.append("")

    filled, total = prose_completeness(s.prose)
    L += [f"## Written answers — {filled}/{total} cells substantively filled",
          "", f"**Rubric marks: ___ / {RUBRIC_TOTAL}** (mark below)", ""]
    for key in sorted(s.prose):
        if key.startswith("_"):
            continue
        v = s.prose[key]
        status = "filled" if v["chars"] > 200 else "**TEMPLATE ONLY**"
        L += [f"### {key} — {v['chars']} chars, {status}", "",
              "```markdown", v["text"][:2000], "```", ""]

    if s.flags:
        L += ["## Flags", ""] + [f"- {f}" for f in s.flags] + [""]
    L += ["## Total", "", f"- Automatic: **{s.auto_score} / {AUTO_TOTAL}**",
          f"- Rubric: **___ / {RUBRIC_TOTAL}**", ""]
    return "\n".join(L)


def load(directory: Path) -> list[Submission]:
    subs = []
    for jf in sorted(directory.glob("submission_*.json")):
        try:
            data = json.loads(jf.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  !! {jf.name}: unreadable JSON ({e})", file=sys.stderr)
            continue
        roll = str(g(data, "student", "roll", default="") or jf.stem.replace("submission_", ""))
        s = Submission(path=jf, roll=roll,
                       name=str(g(data, "student", "name", default="") or ""), data=data)
        cands = list(directory.glob(f"*{roll}*.ipynb")) or list(directory.glob(f"{jf.stem}*.ipynb"))
        if cands:
            s.nb_path = cands[0]
            s.prose = extract_prose(s.nb_path)
        else:
            s.flags.append("no notebook submitted — prose cannot be marked")
        subs.append(s)
    return subs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None, help="report dir")
    ap.add_argument("--csv-only", action="store_true")
    args = ap.parse_args()

    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 2
    out = args.out or args.directory / "report"
    out.mkdir(parents=True, exist_ok=True)

    subs = load(args.directory)
    if not subs:
        print("no submission_*.json found", file=sys.stderr)
        return 1
    for s in subs:
        run_checks(s)

    if not args.csv_only:
        for s in subs:
            (out / f"scorecard_{s.roll}.md").write_text(scorecard(s))

    with (out / "class_summary.csv").open("w", newline="") as f:
        ids = [c.cid for c in subs[0].checks]
        w = csv.writer(f)
        w.writerow(["roll", "name", "device", "auto_score", "auto_max",
                    "prose_filled", "prose_total", *ids, "flags"])
        for s in subs:
            filled, total = prose_completeness(s.prose)
            w.writerow([s.roll, s.name, g(s.data, "env", "device", default=""),
                        s.auto_score, AUTO_TOTAL, filled, total,
                        *[f"{c.earned:g}" for c in s.checks], "; ".join(s.flags)])

    dupes = find_duplicates(subs)
    lines = ["# Integrity report", "",
             f"Submissions: {len(subs)}", ""]
    if dupes:
        lines += ["## Identical measurement vectors", "",
                  "Timing floats carry several digits of run-to-run noise; independent",
                  "runs do not collide. Treat these as copied outputs.", "",
                  "| A | B | values matched |", "|---|---|---:|"]
        lines += [f"| {a} | {b} | {n} |" for a, b, n in dupes]
    else:
        lines.append("No identical measurement vectors found.")
    lines.append("")
    flagged = [s for s in subs if s.flags]
    if flagged:
        lines += ["## Flags", ""]
        lines += [f"- **{s.roll}**: {'; '.join(s.flags)}" for s in flagged]
    (out / "integrity.md").write_text("\n".join(lines) + "\n")

    scores = sorted(s.auto_score for s in subs)
    mean = sum(scores) / len(scores)
    med = scores[len(scores) // 2]
    print(f"graded {len(subs)} submissions -> {out}")
    print(f"  auto score: mean {mean:.1f} / {AUTO_TOTAL}, median {med:.1f}, "
          f"min {scores[0]:.1f}, max {scores[-1]:.1f}")
    print(f"  duplicate pairs: {len(dupes)}")
    print(f"  flagged: {len(flagged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
