#!/usr/bin/env python3
"""
grade_lab13_14.py -- automated marking for Lab 13/14, "Decoding strategies, measured".

    python grade_lab13_14.py <submissions_dir> [-o report_dir] [--csv-only]

WHAT THIS CAN AND CANNOT MARK
-----------------------------
Decoding on a fixed model + prompt is deterministic, so the *magnitudes* here are not
machine-specific the way the timing labs are. What IS gradable is that the lab validates a
model of decoding whose predictions are *ordinal* and hold for anyone:

    greedy run1 == run2                 argmax is deterministic; no randomness to diverge
    greedy first token == argmax        greedy IS the argmax at step 0
    P(top) falls as temperature rises   softmax(l/T): the top logit's lead shrinks with T
    diversity rises as temperature rises a flatter distribution spreads the draws
    nucleus(peaked) < nucleus(flat)     top-p adapts its set size to the distribution
    top-k(peaked) == top-k(flat) == k   top-k does not adapt -- it keeps k regardless
    top-p kept mass >= p, set < vocab   the nucleus reaches p by truncating a real tail

The sampling part (diversity) is seeded from the student's roll number, so two honest
submissions never share a diversity vector -- that vector is the plagiarism fingerprint.

This marks those relationships between a student's OWN numbers (40) and extracts their
prose for rubric marking (60). It never compares magnitudes to a reference.

DELIBERATELY NOT AUTO-CHECKED
-----------------------------
- Part 4 (beam vs greedy). Beam is a heuristic and does NOT reliably beat greedy on total
  log-probability -- that surprise is the point of the part, and it is rubric-marked from
  the student's explanation, not scored here.
- The exact greedy text / logprobs. On the default prompt they are identical for everyone,
  so they carry no marks; only the relationships above do.
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
    dist = R.get("dist", {}); temp = R.get("temperature", {}); tr = R.get("truncation", {})

    c = Check("A1", "Provenance: roll, name, torch, transformers", 3)
    have = [bool(s.roll), bool(d.get("name")), bool(env.get("torch")), bool(env.get("transformers"))]
    c.earned = round(c.marks*sum(have)/len(have), 2); c.ok = all(have); c.detail = f"{sum(have)}/4"; add(c)

    c = Check("A2", "Part 1 records the distribution and greedy metrics", 4)
    need = ["vocab_size", "top50_mass", "greedy_ids_run1", "greedy_ids_run2", "greedy_first_id", "argmax_id"]
    have = sum(1 for k in need if dist.get(k) is not None)
    c.earned = round(c.marks*have/len(need), 2); c.ok = have == len(need); c.detail = f"{have}/{len(need)} fields"; add(c)

    c = Check("A3", "Greedy is deterministic (run1 == run2)", 6)
    r1, r2 = dist.get("greedy_ids_run1"), dist.get("greedy_ids_run2")
    c.ok = isinstance(r1, list) and r1 and r1 == r2
    c.earned = c.marks if c.ok else 0.0
    c.detail = "identical" if c.ok else "runs differ or missing"; add(c)

    c = Check("A4", "Greedy's first token is the argmax", 3)
    gf, am = dist.get("greedy_first_id"), dist.get("argmax_id")
    c.ok = gf is not None and gf == am; c.earned = c.marks if c.ok else 0.0
    c.detail = f"first={gf}, argmax={am}"; add(c)

    c = Check("A5", "P(top token) falls monotonically as temperature rises", 7)
    Ts = [num(x) for x in (temp.get("Ts") or [])]
    pt = [num(x) for x in (temp.get("p_top") or [])]
    if len(pt) >= 2 and all(v is not None for v in pt) and Ts == sorted(Ts):
        drops = sum(1 for a, b in zip(pt, pt[1:]) if b < a)
        c.earned = round(c.marks*drops/(len(pt)-1), 2); c.ok = drops == len(pt)-1
        c.detail = f"{drops}/{len(pt)-1} steps fall: {pt}"
    else: c.detail = "need >=2 p_top with increasing Ts"
    add(c)

    c = Check("A6", "Sample diversity rises with temperature", 6)
    dr = [num(x) for x in (temp.get("distinct_ratio") or [])]
    if len(dr) >= 2 and all(v is not None for v in dr):
        rose = dr[-1] > dr[0]
        strict = all(b > a for a, b in zip(dr, dr[1:]))
        c.ok = strict; c.earned = c.marks if strict else (c.marks*0.5 if rose else 0.0)
        c.detail = f"{'strictly ' if strict else ''}{'rising' if rose else 'not rising'}: {dr}"
    else: c.detail = "need >=2 distinct_ratio points"
    add(c)

    c = Check("A7", "Nucleus adapts (peaked < flat); top-k is fixed", 7)
    npk, nfl = num(tr.get("nucleus_peaked")), num(tr.get("nucleus_flat"))
    tpk, tfl, k = num(tr.get("topk_peaked")), num(tr.get("topk_flat")), num(tr.get("topk_k"))
    adapts = npk is not None and nfl is not None and npk < nfl
    fixed = None not in (tpk, tfl, k) and tpk == tfl == k
    c.ok = adapts and fixed; c.earned = round(c.marks*(bool(adapts)+bool(fixed))/2, 2)
    c.detail = f"nucleus peaked={npk} flat={nfl} (adapts={adapts}); top-k fixed={fixed}"; add(c)

    c = Check("A8", "Top-p reaches p by truncating a real tail", 4)
    kept, p, nfl, vocab = num(tr.get("topp_kept_mass")), num(tr.get("nucleus_p")), \
                          num(tr.get("nucleus_flat")), num(dist.get("vocab_size"))
    reaches = kept is not None and p is not None and kept >= p - 1e-4
    truncates = nfl is not None and vocab is not None and 1 <= nfl < vocab
    c.ok = reaches and truncates; c.earned = round(c.marks*(bool(reaches)+bool(truncates))/2, 2)
    c.detail = f"kept={kept} >= p={p} ({reaches}); nucleus {nfl} < vocab {vocab} ({truncates})"; add(c)

PLACEHOLDER = "your "  # template cells contain "(your ... here)"
def extract_prose(nb_path: Path) -> dict:
    try: nb = json.loads(nb_path.read_text())
    except Exception: return {}
    out = {}; i = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown": continue
        txt = "".join(cell.get("source", []))
        if "\U0001F4DD" in txt:  # the 📝 prompt marker
            i += 1
            body = [l for l in txt.splitlines() if l.strip() and "\U0001F4DD" not in l
                    and not (l.strip().startswith("*(") and l.strip().endswith(")*"))
                    and PLACEHOLDER not in l]
            out[f"prompt_{i}"] = "\n".join(body).strip()
    return out

def prose_completeness(prose: dict):
    if not prose: return 0, 0
    filled = sum(1 for v in prose.values() if len(v) > 20)
    return filled, len(prose)

def fingerprint(s: Submission):
    R = s.data.get("results", {})
    vals = list(R.get("temperature", {}).get("distinct_ratio") or [])       # seeded from roll -> per-student
    vals += list(R.get("temperature", {}).get("p_top") or [])
    tr = R.get("truncation", {}); vals += [tr.get("nucleus_peaked"), tr.get("nucleus_flat")]
    vals.append(R.get("beam", {}).get("beam_logprob"))
    return tuple(vals)

def scorecard(s: Submission) -> str:
    L = [f"{'='*60}", f"Lab 13/14  --  {s.roll}  ({s.data.get('name','?')})", "="*60,
         f"Prompt: {s.data.get('env',{}).get('prompt','?')!r}   seed: {s.data.get('env',{}).get('seed','?')}", "",
         f"AUTOMATIC  {s.auto:.1f} / {AUTO_TOTAL}", "-"*60]
    for c in s.checks:
        L.append(f"  [{'x' if c.ok else ' '}] {c.id} {c.desc:<48} {c.earned:4.1f}/{c.marks}")
        if c.detail: L.append(f"        {c.detail}")
    f, t = prose_completeness(s.prose)
    L += ["", f"RUBRIC (human)  -- / {RUBRIC_TOTAL}", "-"*60,
          f"  {f}/{t} 📝 cells filled" + (f"   ({t-f} left as template)" if t and f < t else ""),
          "  Rubric weighting: Part 2's temperature-ratio explanation and Part 4's",
          "  'why beam can lose to greedy' paragraph carry the most marks.", ""]
    for k, v in s.prose.items():
        L.append(f"  --- {k} ---")
        L.append("      " + (v.replace("\n", "\n      ") if v else "(EMPTY / template)"))
    if s.flags: L += ["", "FLAGS: " + "; ".join(s.flags)]
    return "\n".join(L)

def load(directory: Path):
    subs = []
    for jf in sorted(directory.glob("submission_lab13_14_*.json")):
        try: data = json.loads(jf.read_text())
        except Exception as e:
            print(f"  !! {jf.name}: bad JSON ({e})", file=sys.stderr); continue
        roll = str(data.get("roll") or jf.stem.replace("submission_lab13_14_", ""))
        nb = next(iter(directory.glob(f"*{roll}*.ipynb")), None)
        subs.append(Submission(roll=roll, data=data, nb_path=nb))
    return subs

def main() -> int:
    ap = argparse.ArgumentParser(description="Mark Lab 13/14 submissions.")
    ap.add_argument("directory", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--csv-only", action="store_true")
    a = ap.parse_args()
    subs = load(a.directory)
    if not subs:
        print(f"No submission_lab13_14_*.json in {a.directory}", file=sys.stderr); return 1

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
        w = csv.writer(f); w.writerow(["roll", "name", "auto", "auto_total", "prose_filled", "prose_total", "flags"])
        for s in subs:
            fl, tt = prose_completeness(s.prose)
            w.writerow([s.roll, s.data.get("name", ""), s.auto, AUTO_TOTAL, fl, tt, "; ".join(s.flags)])
    print(f"marked {len(subs)} submission(s)")
    for s in subs:
        print(f"  {s.roll:12s} auto {s.auto:5.1f}/{AUTO_TOTAL}" + (f"   FLAG: {'; '.join(s.flags)}" if s.flags else ""))
    print(f"wrote {out/'marks.csv'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
