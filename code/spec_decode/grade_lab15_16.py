#!/usr/bin/env python3
"""
grade_lab15_16.py -- automated marking for Lab 15/16, "Speculative decoding, measured".

    python grade_lab15_16.py <submissions_dir> [-o report_dir] [--csv-only]

Same principle as the other DS635 labs: every check tests a *relationship between the student's
own numbers*, never a magnitude against a reference. The dummy-mechanics numbers are seeded from
the roll number (so two honest submissions never share a vector); the real-stitch numbers are the
greedy-identity and pass-count invariants, which hold on any machine.

    accept empirical ~ min(1, p/q)      the ratio test is the acceptance probability
    emitted dist ~ p (small TV)         the residual makes the output exactly p
    broken (resample p) -> large TV     the silent bug a distribution test catches
    vectorised cut == scalar cut        the vectorised verify agrees with the loop
    spec-greedy == plain greedy         the stitch is lossless, token for token
    target passes < tokens              the stitch verifies many positions per weight read
    accept lengths in [0, k]            a block yields 1..k+1 tokens, never 0

40 automatic + 60 rubric (the 📝 prose, extracted for a human).
"""
import argparse, csv, json, sys
from dataclasses import dataclass, field
from pathlib import Path

AUTO_TOTAL, RUBRIC_TOTAL = 40, 60

@dataclass
class Check:
    id: str; desc: str; marks: float
    earned: float = 0.0; ok: bool = False; detail: str = ""

@dataclass
class Submission:
    roll: str; data: dict; nb_path: Path | None = None
    checks: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    prose: dict = field(default_factory=dict)
    @property
    def auto(self): return round(sum(c.earned for c in self.checks), 2)

def num(x):
    try: v = float(x)
    except (TypeError, ValueError): return None
    return v if v == v and abs(v) != float("inf") else None

def run_checks(s: Submission):
    d = s.data; R = d.get("results", {}); env = d.get("env", {}); add = s.checks.append
    acc = R.get("accept", {}); emit = R.get("emit", {}); brk = R.get("broken", {})
    vec = R.get("vector", {}); st = R.get("stitch", {})

    c = Check("A1", "Provenance: roll, name, torch, transformers", 3)
    have = [bool(s.roll), bool(d.get("name")), bool(env.get("torch")), bool(env.get("transformers"))]
    c.earned = round(c.marks*sum(have)/len(have), 2); c.ok = all(have); c.detail = f"{sum(have)}/4"; add(c)

    c = Check("A2", "Accept rate tracks min(1, p/q) at each position", 5)
    e = [num(x) for x in (acc.get("empirical") or [])]
    t = [num(x) for x in (acc.get("theoretical") or [])]
    if e and t and len(e) == len(t) and all(v is not None for v in e+t):
        ok = [abs(a-b) < 0.03 for a, b in zip(e, t)]
        c.earned = round(c.marks*sum(ok)/len(ok), 2); c.ok = all(ok)
        c.detail = f"{sum(ok)}/{len(ok)} positions within 0.03"
    else: c.detail = "missing accept rates"
    add(c)

    c = Check("A3", "Emitted distribution matches p (residual correct)", 8)
    tv = num(emit.get("tv_distance"))
    if tv is not None:
        c.ok = tv < 0.02; c.earned = c.marks if c.ok else 0.0
        c.detail = f"TV(emitted, p) = {tv}"
    else: c.detail = "missing emit TV"
    add(c)

    c = Check("A4", "Broken control (resample p) is caught by TV, not acceptance", 4)
    bt = num(brk.get("tv_distance"))
    if bt is not None and tv is not None:
        c.ok = bt > max(0.03, 3*tv); c.earned = c.marks if c.ok else 0.0
        c.detail = f"broken TV {bt} vs correct TV {tv}"
    else: c.detail = "missing broken TV"
    add(c)

    c = Check("A5", "Vectorised verify agrees with the scalar cut", 5)
    ag = num(vec.get("agreement"))
    if ag is not None:
        c.ok = ag >= 0.999; c.earned = c.marks if c.ok else round(c.marks*max(0.0, ag-0.9)/0.1, 2)
        c.detail = f"agreement {ag}"
    else: c.detail = "missing agreement"
    add(c)

    c = Check("A6", "Speculative-greedy output == plain greedy (lossless)", 8)
    idg = st.get("identical_to_greedy")
    c.ok = idg is True; c.earned = c.marks if c.ok else 0.0
    c.detail = f"identical_to_greedy = {idg}"; add(c)

    c = Check("A7", "Fewer target passes than tokens generated", 4)
    tp, nt = num(st.get("target_passes")), num(st.get("n_tokens"))
    if tp is not None and nt is not None:
        c.ok = tp < nt; c.earned = c.marks if c.ok else 0.0
        c.detail = f"{tp:.0f} passes for {nt:.0f} tokens"
    else: c.detail = "missing pass counts"
    add(c)

    c = Check("A8", "Accepted-run lengths in [0, k] (block yields 1..k+1)", 3)
    al = st.get("accept_lengths"); kk = num(st.get("k"))
    if isinstance(al, list) and al and kk is not None:
        c.ok = all(isinstance(a, int) and 0 <= a <= kk for a in al)
        c.earned = c.marks if c.ok else 0.0
        c.detail = f"{len(al)} blocks, lengths in [{min(al)}, {max(al)}], k={kk:.0f}"
    else: c.detail = "missing accept_lengths"
    add(c)

PLACEHOLDER = "your "
def extract_prose(nb_path: Path) -> dict:
    try: nb = json.loads(nb_path.read_text())
    except Exception: return {}
    out = {}; i = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown": continue
        txt = "".join(cell.get("source", []))
        if "\U0001F4DD" in txt:
            i += 1
            body = [l for l in txt.splitlines() if l.strip() and "\U0001F4DD" not in l
                    and not (l.strip().startswith("*(") and l.strip().endswith(")*"))
                    and PLACEHOLDER not in l]
            out[f"prompt_{i}"] = "\n".join(body).strip()
    return out

def prose_completeness(prose: dict):
    if not prose: return 0, 0
    return sum(1 for v in prose.values() if len(v) > 20), len(prose)

def fingerprint(s: Submission):
    R = s.data.get("results", {})
    vals = list(R.get("emit", {}).get("empirical") or [])          # seeded from roll -> per-student
    vals += list(R.get("accept", {}).get("empirical") or [])
    vals.append(R.get("vector", {}).get("agreement"))
    return tuple(vals)

def scorecard(s: Submission) -> str:
    L = [f"{'='*60}", f"Lab 15/16  --  {s.roll}  ({s.data.get('name','?')})", "="*60,
         f"Seed: {s.data.get('env',{}).get('seed','?')}   models: "
         f"{s.data.get('env',{}).get('target','?')} <- {s.data.get('env',{}).get('draft','?')}", "",
         f"AUTOMATIC  {s.auto:.1f} / {AUTO_TOTAL}", "-"*60]
    for c in s.checks:
        L.append(f"  [{'x' if c.ok else ' '}] {c.id} {c.desc:<48} {c.earned:4.1f}/{c.marks}")
        if c.detail: L.append(f"        {c.detail}")
    f, t = prose_completeness(s.prose)
    L += ["", f"RUBRIC (human)  -- / {RUBRIC_TOTAL}", "-"*60,
          f"  {f}/{t} 📝 cells filled" + (f"   ({t-f} left as template)" if t and f < t else ""),
          "  Heaviest: Part 2's 'why the residual, not p' paragraph and Part 4's",
          "  'acceptance is not speedup' write-up.", ""]
    for k, v in s.prose.items():
        L.append(f"  --- {k} ---")
        L.append("      " + (v.replace("\n", "\n      ") if v else "(EMPTY / template)"))
    if s.flags: L += ["", "FLAGS: " + "; ".join(s.flags)]
    return "\n".join(L)

def load(directory: Path):
    subs = []
    for jf in sorted(directory.glob("submission_lab15_16_*.json")):
        try: data = json.loads(jf.read_text())
        except Exception as e:
            print(f"  !! {jf.name}: bad JSON ({e})", file=sys.stderr); continue
        roll = str(data.get("roll") or jf.stem.replace("submission_lab15_16_", ""))
        nb = next(iter(directory.glob(f"*{roll}*.ipynb")), None)
        subs.append(Submission(roll=roll, data=data, nb_path=nb))
    return subs

def main() -> int:
    ap = argparse.ArgumentParser(description="Mark Lab 15/16 submissions.")
    ap.add_argument("directory", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--csv-only", action="store_true")
    a = ap.parse_args()
    subs = load(a.directory)
    if not subs:
        print(f"No submission_lab15_16_*.json in {a.directory}", file=sys.stderr); return 1
    seen = {}
    for s in subs:
        run_checks(s)
        if s.nb_path: s.prose = extract_prose(s.nb_path)
        fp = fingerprint(s)
        if fp in seen: s.flags.append(f"identical measurements to {seen[fp]}")
        else: seen[fp] = s.roll
    out = a.out or a.directory
    out.mkdir(exist_ok=True, parents=True)
    if not a.csv_only:
        for s in subs: (out/f"report_{s.roll}.txt").write_text(scorecard(s))
    with open(out/"marks.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["roll","name","auto","auto_total","prose_filled","prose_total","flags"])
        for s in subs:
            fl, tt = prose_completeness(s.prose)
            w.writerow([s.roll, s.data.get("name",""), s.auto, AUTO_TOTAL, fl, tt, "; ".join(s.flags)])
    print(f"marked {len(subs)} submission(s)")
    for s in subs:
        print(f"  {s.roll:12s} auto {s.auto:5.1f}/{AUTO_TOTAL}" + (f"   FLAG: {'; '.join(s.flags)}" if s.flags else ""))
    print(f"wrote {out/'marks.csv'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
