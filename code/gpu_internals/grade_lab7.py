#!/usr/bin/env python3
"""
grade_lab7.py -- automated marking for Lab 7, "The concurrency budget".

    python grade_lab7.py <submissions_dir> [-o report_dir] [--csv-only]

WHAT THIS CAN AND CANNOT MARK
-----------------------------
Same principle as grade_submissions.py for Lab 5/6: every number in this lab is
machine-dependent by design. A Colab T4, an RX 6700M and an RTX 3060 disagree on
every value, and all three are correct. So there is no answer key.

What IS gradable is that the lab validates a physical model whose predictions are
*ordinal*, and hold on any GPU:

    latency grows with footprint          there is a memory hierarchy
    one block cannot saturate a GPU       bandwidth needs concurrency
    bandwidth x latency ~ bytes in flight Little's Law is not a GPU fact
    fixed work per block gives a          block scheduling is quantised
      staircase with flat treads

This script marks those relationships between a student's OWN numbers (35 marks)
and extracts their prose for rubric marking (65 marks). It never compares a
student's magnitudes against a reference.

TRAPS DELIBERATELY NOT CHECKED
------------------------------
- **The number of latency plateaus.** Measured on the course laptop: four visible
  levels, but the L2 and Infinity-Cache steps are indistinguishable in latency and
  a student who reports three is reading their data correctly. On a T4 the shape
  is different again. Asserting a count would fail students for being right.
- **Whether the step period equals `multi_processor_count`.** On the course RX 6700M
  the period is 18 and `multi_processor_count` is 18 -- but the card has 36 CUs, and
  a student arguing the useful number is 36 is making the better argument. A9 only
  checks the period is within a factor of two of the reported count, so 18 and 36
  both pass and a nonsense value like 5 does not.
- **How close the Little's Law ratio lands.** The identity holds; the residue is
  prefetching, vector width and per-block memory-level parallelism, and explaining
  it is Task 3.2 -- worth 19 rubric marks precisely because the ratio itself is not
  worth automatic ones. A7 accepts anything inside a factor of eight either way.
- **The direction of the last staircase step.** Boost-clock decay over a long sweep
  can make a late tread slightly cheaper than an early one. Only the first tread is
  checked for flatness.

A student is never penalised for what their hardware did.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

AUTO_TOTAL = 35
RUBRIC_TOTAL = 65

REQUIRED_KEYS = [
    "latency_footprint_kb", "latency_ns", "latency_memory_ns",
    "bw_blocks", "bw_gbs", "bw_peak_gbs", "bw_one_block_gbs",
    "littles_law_bytes_needed", "knee_bytes_measured", "littles_law_ratio",
    "staircase_blocks", "staircase_ms", "staircase_period",
]

PLACEHOLDERS = ("your prediction:", "your answer:", "*your prediction*", "*your answer*")


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
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def num(x):
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
    d = s.data
    R = d.get("results", {})
    env = d.get("env", {})
    ans = d.get("answers", {})
    add = s.checks.append

    # --- A1 provenance -----------------------------------------------------
    c = Check("A1", "Provenance: device, torch, Triton, identity", 3)
    have = [bool(env.get("device")), bool(env.get("torch")),
            bool(env.get("triton")), bool(s.roll)]
    c.earned = round(c.marks * sum(have) / len(have), 2)
    c.ok = all(have)
    c.detail = f"{sum(have)}/4 present"
    add(c)

    # --- A2 completeness ---------------------------------------------------
    c = Check("A2", "All measurements recorded", 4)
    present = [k for k in REQUIRED_KEYS if R.get(k) is not None]
    c.earned = round(c.marks * len(present) / len(REQUIRED_KEYS), 2)
    c.ok = len(present) == len(REQUIRED_KEYS)
    missing = [k for k in REQUIRED_KEYS if k not in present]
    c.detail = f"{len(present)}/{len(REQUIRED_KEYS)}" + (f"; missing {missing}" if missing else "")
    add(c)

    # --- A3 latency rises with footprint -----------------------------------
    # Allow 10% slack: boost-clock drift makes a strictly monotone sequence rare.
    c = Check("A3", "Latency never falls as footprint grows (10% slack)", 4)
    lat = [num(v) for v in (R.get("latency_ns") or [])]
    lat = [v for v in lat if v is not None]
    if len(lat) >= 4:
        drops = sum(1 for a, b in zip(lat, lat[1:]) if b < 0.90 * a)
        c.ok = drops == 0
        c.earned = c.marks if c.ok else max(0.0, c.marks * (1 - drops / len(lat)))
        c.detail = f"{len(lat)} points, {drops} drop(s) beyond noise"
    else:
        c.detail = "fewer than 4 latency points recorded"
    add(c)

    # --- A4 the hierarchy is visible at all --------------------------------
    c = Check("A4", "Slowest latency >= 2x the fastest", 3)
    if lat:
        ratio = max(lat) / min(lat)
        c.ok = ratio >= 2.0
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"{min(lat):.0f} -> {max(lat):.0f} ns  ({ratio:.1f}x)"
    else:
        c.detail = "no latency data"
    add(c)

    # --- A5 the chase produced a believable number -------------------------
    c = Check("A5", "Memory latency in the 100-2000 ns sanity band", 3)
    lm = num(R.get("latency_memory_ns"))
    if lm is not None:
        c.ok = 100.0 <= lm <= 2000.0
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"{lm:.0f} ns" + ("" if c.ok else "  <- chase likely broken (stride? dead loop?)")
    else:
        c.detail = "not recorded"
    add(c)

    # --- A6 one block cannot saturate a GPU --------------------------------
    c = Check("A6", "Peak bandwidth >= 3x the one-block bandwidth", 4)
    peak, one = num(R.get("bw_peak_gbs")), num(R.get("bw_one_block_gbs"))
    if peak and one:
        ratio = peak / one
        c.ok = ratio >= 3.0
        c.earned = c.marks if c.ok else round(c.marks * min(1.0, ratio / 3.0), 2)
        c.detail = f"{one:.0f} -> {peak:.0f} GB/s  ({ratio:.1f}x)"
    else:
        c.detail = "bandwidth sweep incomplete"
    add(c)

    # --- A7 Little's Law ---------------------------------------------------
    c = Check("A7", "Little's Law ratio within 8x either way", 5)
    ratio = num(R.get("littles_law_ratio"))
    if ratio and ratio > 0:
        c.ok = 0.125 <= ratio <= 8.0
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"measured/predicted = {ratio:.2f}x"
        if c.ok and not (0.5 <= ratio <= 2.0):
            s.flags.append(f"Little's Law ratio {ratio:.2f}x is loose but inside the band "
                           f"-- check Task 3.2 explains the direction of the error.")
    else:
        c.detail = "not recorded"
    add(c)

    # --- A8 the staircase --------------------------------------------------
    c = Check("A8", "A staircase exists, and its first tread is flat", 5)
    ms = [num(v) for v in (R.get("staircase_ms") or [])]
    ms = [v for v in ms if v is not None]
    period = num(R.get("staircase_period"))
    plateau = num(R.get("staircase_first_plateau"))
    if len(ms) >= 8:
        has_step = any(b > 1.10 * a for a, b in zip(ms, ms[1:]))
        bits = [has_step]
        detail = ["step >=10%: " + ("yes" if has_step else "NO")]
        if plateau and plateau >= 3:
            tread = ms[1:int(plateau)]              # skip block 1: launch overhead dominates
            flat = (max(tread) - min(tread)) / max(tread) < 0.10 if tread else False
            bits.append(flat)
            detail.append(f"first tread spread {100*(max(tread)-min(tread))/max(tread):.1f}%")
        c.earned = round(c.marks * sum(bits) / len(bits), 2)
        c.ok = all(bits)
        c.detail = "; ".join(detail)
    else:
        c.detail = "staircase sweep too short"
    add(c)

    # --- A9 the period is a plausible hardware number ----------------------
    c = Check("A9", "Step period within a factor of 2 of the reported unit count", 2)
    units = num(env.get("multi_processor_count")) or num(R.get("units_reported"))
    if period and units:
        r = period / units
        c.ok = 0.5 <= r <= 2.0
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"period {period:.0f} vs units {units:.0f}  ({r:.2f}x)"
    elif period:
        c.detail = "unit count not recorded"
    else:
        c.detail = "no period detected -- see flags"
        s.flags.append("No staircase period detected. If the student explains a flat "
                       "curve honestly in Task 4.2, award A8/A9 manually.")
    add(c)

    # --- A10 the student read their own data -------------------------------
    c = Check("A10", "Short answers agree with the student's own measurements", 2)
    pairs = [
        ("bw_peak_gbs", R.get("bw_peak_gbs")),
        ("knee_blocks_measured", R.get("knee_blocks_measured")),
        ("latency_memory_ns", R.get("latency_memory_ns")),
        ("staircase_period_blocks", R.get("staircase_period")),
        ("littles_law_ratio", R.get("littles_law_ratio")),
    ]
    checked = [(k, v) for k, v in pairs if v is not None and ans.get(k) is not None]
    if checked:
        agree = sum(1 for k, v in checked if close(ans[k], v, tol=0.10))
        c.earned = round(c.marks * agree / len(checked), 2)
        c.ok = agree == len(checked)
        bad = [k for k, v in checked if not close(ans[k], v, tol=0.10)]
        c.detail = f"{agree}/{len(checked)} agree" + (f"; mismatched {bad}" if bad else "")
    else:
        c.detail = "answers block empty -- nothing to cross-check"
        s.flags.append("ANSWERS dict was left unfilled; the notebook's own export cell "
                       "warns about this, so the submission was exported knowingly incomplete.")
    add(c)


# --------------------------------------------------------------------------
# prose
# --------------------------------------------------------------------------

def extract_prose(nb_path: Path) -> dict:
    """Pull the tagged 📝 / 🔮 cells out of the executed notebook."""
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
        body = "".join(cell.get("source", []))
        # Strip the prompt so only the student's own writing is measured.
        lines = [l for l in body.split("\n")
                 if l.strip() and not any(ph in l.lower() for ph in PLACEHOLDERS)]
        out[f"{task or 'untagged'}:{kind}"] = {
            "chars": len("\n".join(lines).strip()), "text": body.strip()}
    return out


def prose_completeness(prose: dict) -> tuple[int, int]:
    """(filled, total) -- a cell counts as filled if it grew beyond its template.

    The templates in this lab are long (the prompts carry the marks breakdown), so
    the threshold is on GROWTH: the shipped notebook is measured once at import and
    anything materially longer than its own prompt counts as written.
    """
    cells = {k: v for k, v in prose.items() if not k.startswith("_")}
    if not cells:
        return 0, 0
    filled = sum(1 for v in cells.values() if v["chars"] > TEMPLATE_CHARS.get(_key(v), 0) + 200)
    return filled, len(cells)


TEMPLATE_CHARS: dict = {}


def _key(v):
    return v["text"][:60]


def learn_templates(blank_nb: Path) -> None:
    """Record the length of every prompt in the pristine notebook.

    Without this a long prompt looks like a long answer. Point it at the notebook
    as shipped: `--blank docs/labs/Lab7_concurrency_budget.ipynb`.
    """
    for k, v in extract_prose(blank_nb).items():
        if not k.startswith("_"):
            TEMPLATE_CHARS[_key(v)] = v["chars"]


# --------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------

def float_fingerprint(sub: Submission) -> tuple:
    """The measured floats, in a stable order. Two students matching is ~impossible."""
    vals = []
    R = sub.data.get("results", {})
    for k in sorted(R):
        v = R[k]
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            vals.append(round(float(v), 6))
        elif isinstance(v, list):
            vals += [round(float(x), 6) for x in v
                     if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return tuple(vals)


def find_duplicates(subs: list[Submission]) -> list[tuple[str, str, int]]:
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
    env = s.data.get("env", {})
    L = [f"# Lab 7 scorecard — {s.roll}" + (f" ({s.name})" if s.name else ""), ""]
    L += [f"- **Device:** {env.get('device','?')}",
          f"- **Backend:** torch {env.get('torch','?')} / {env.get('backend','?')} / "
          f"triton {env.get('triton','?')}",
          f"- **Units reported:** {env.get('multi_processor_count','?')}  "
          f"**warp:** {env.get('warp_size','?')}",
          f"- **Safe mode:** {env.get('safe_mode','?')}  "
          f"(on battery the sweeps are scaled down; not penalised)",
          f"- **Run:** {env.get('started_utc','?')} → {env.get('finished_utc','?')}",
          f"- **Notebook:** {s.nb_path.name if s.nb_path else '**MISSING**'}", ""]
    L += [f"## Automatic checks — {s.auto_score} / {AUTO_TOTAL}", "",
          "| ID | Check | Marks | Result | Detail |", "|---|---|---:|---|---|"]
    for c in s.checks:
        mark = "PASS" if c.ok else ("PARTIAL" if 0 < c.earned < c.marks else "FAIL")
        L.append(f"| {c.cid} | {c.label} | {c.earned:g}/{c.marks:g} | {mark} | {c.detail} |")
    L.append("")

    filled, total = prose_completeness(s.prose)
    L += [f"## Written answers — {filled}/{total} cells substantively filled", "",
          "Suggested rubric split: Task 3.2 is worth 19, Task 4.2 12, Task 4.3 13, "
          "Task 1.2 9, Task 5.1 10, the rest 2 marks per prediction.", "",
          f"**Rubric marks: ___ / {RUBRIC_TOTAL}** (mark below)", ""]
    for key in sorted(s.prose):
        if key.startswith("_"):
            continue
        v = s.prose[key]
        thresh = TEMPLATE_CHARS.get(_key(v), 0) + 200
        status = "filled" if v["chars"] > thresh else "**TEMPLATE ONLY**"
        L += [f"### {key} — {v['chars']} chars, {status}", "",
              "```markdown", v["text"][:2000], "```", ""]

    if s.flags:
        L += ["## Flags", ""] + [f"- {f}" for f in s.flags] + [""]
    L += ["## Total", "", f"- Automatic: **{s.auto_score} / {AUTO_TOTAL}**",
          f"- Rubric: **___ / {RUBRIC_TOTAL}**", ""]
    return "\n".join(L)


def load(directory: Path) -> list[Submission]:
    subs = []
    for jf in sorted(directory.glob("submission_lab7_*.json")):
        try:
            data = json.loads(jf.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  !! {jf.name}: unreadable JSON ({e})", file=sys.stderr)
            continue
        roll = str(g(data, "student", "roll", default="")
                   or jf.stem.replace("submission_lab7_", ""))
        s = Submission(path=jf, roll=roll,
                       name=str(g(data, "student", "name", default="") or ""), data=data)
        cands = list(directory.glob(f"*{roll}*.ipynb"))
        if cands:
            s.nb_path = cands[0]
            s.prose = extract_prose(s.nb_path)
        else:
            s.flags.append("No notebook found for this roll number. The JSON alone "
                           "cannot be marked -- 65 of 100 marks live in the prose.")
        subs.append(s)
    return subs


def main() -> int:
    ap = argparse.ArgumentParser(description="Mark Lab 7 submissions.")
    ap.add_argument("directory", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write per-student scorecards here (default: <dir>/reports)")
    ap.add_argument("--blank", type=Path,
                    default=Path("docs/labs/Lab7_concurrency_budget.ipynb"),
                    help="the notebook as shipped, to measure prompt lengths against")
    ap.add_argument("--csv-only", action="store_true")
    a = ap.parse_args()

    if a.blank.exists():
        learn_templates(a.blank)
    else:
        print(f"  !! blank notebook not found at {a.blank}; prose completeness will be "
              f"over-generous. Pass --blank.", file=sys.stderr)

    subs = load(a.directory)
    if not subs:
        print(f"No submission_lab7_*.json found in {a.directory}", file=sys.stderr)
        return 1
    for s in subs:
        run_checks(s)

    for r1, r2, n in find_duplicates(subs):
        msg = f"IDENTICAL measurement vector ({n} floats) shared with {r2}"
        for s in subs:
            if s.roll == r1:
                s.flags.append(msg)
            elif s.roll == r2:
                s.flags.append(f"IDENTICAL measurement vector ({n} floats) shared with {r1}")

    out = a.out or (a.directory / "reports")
    if not a.csv_only:
        out.mkdir(parents=True, exist_ok=True)
        for s in subs:
            (out / f"scorecard_{s.roll}.md").write_text(scorecard(s))

    csv_path = (out if not a.csv_only else a.directory) / "lab7_marks.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roll", "name", "device", "auto", "auto_total",
                    "prose_filled", "prose_total", "flags"])
        for s in subs:
            filled, total = prose_completeness(s.prose)
            w.writerow([s.roll, s.name, g(s.data, "env", "device", default=""),
                        s.auto_score, AUTO_TOTAL, filled, total, " | ".join(s.flags)])

    print(f"marked {len(subs)} submission(s)")
    for s in subs:
        filled, total = prose_completeness(s.prose)
        print(f"  {s.roll:12s} auto {s.auto_score:5.1f}/{AUTO_TOTAL}   "
              f"prose {filled}/{total}" + ("   FLAGS" if s.flags else ""))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
