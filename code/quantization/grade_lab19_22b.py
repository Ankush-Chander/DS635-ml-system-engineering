#!/usr/bin/env python3
"""
grade_lab19_22b.py -- automated marking for Lab 19-22b, "Quantization error, measured".

    python grade_lab19_22b.py <submissions_dir> [-o report_dir] [--csv-only]

Same principle as the other DS635 labs: every check tests a *relationship between the student's
own numbers*, never a magnitude against a reference. The signals, the data split and the model
are seeded from the roll number, so two honest submissions never share a measurement vector.

    MSE ~ S^2 / 12 at 4-8 bits                  rounding error spread evenly over one step
    SNR rises ~6 dB per bit, sine and noise     one bit halves S, a quarter of the error power
    sine SNR > noise SNR at every width         the sine fills its range, the noise does not
    outlier drop ~ 20 log10(range ratio)        one value sets S for everything that shares it
    blocks of 64 recover the drop               a smaller group confines the outlier
    8-bit accuracy within 1% of float           the model's error budget is far above 8-bit noise
    accuracy collapses at 1-2 bits              too few levels to separate the weights
    per-channel: <= 2^b levels per column       each output column gets its own S
    per-channel weight SNR > per-tensor         the widest column no longer sets everyone's S

40 automatic + 60 rubric (the 📝 prose, extracted for a human).
"""
import argparse, csv, json, math, sys
from dataclasses import dataclass, field
from pathlib import Path

AUTO_TOTAL, RUBRIC_TOTAL = 40, 60
LAB = "lab19_22b"

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

def all_num(*xs): return all(x is not None for x in xs)

def frac(c: Check, oks):
    c.earned = round(c.marks*sum(oks)/len(oks), 2) if oks else 0.0; c.ok = bool(oks) and all(oks)

def run_checks(s: Submission):
    d = s.data; R = d.get("results", {}); env = d.get("env", {}); add = s.checks.append
    sw = R.get("sweep", {}); ol = R.get("outlier", {}); m = R.get("model", {})
    rows = {name: {int(r.get("bits", -1)): r for r in sw.get(name, {}).get("rows", [])} for name in ("sine", "noise")}

    c = Check("A1", "Provenance: roll, name, numpy, sklearn", 2)
    have = [bool(s.roll), bool(d.get("name")), bool(env.get("numpy")), bool(env.get("sklearn"))]
    frac(c, have); c.detail = f"{sum(have)}/4"; add(c)

    # --- Part 1: the error playground ---
    c = Check("B1", "MSE / (S^2/12) in [0.8, 1.25] at 4-8 bits, sine and noise", 6)
    ratios = []
    for name in ("sine", "noise"):
        for b in range(4, 9):
            r = rows[name].get(b, {}); mse, s12 = num(r.get("mse")), num(r.get("s2_over_12"))
            ratios.append(mse / s12 if all_num(mse, s12) and s12 > 0 else None)
    frac(c, [x is not None and 0.8 <= x <= 1.25 for x in ratios])
    c.detail = f"{sum(1 for x in ratios if x is not None and 0.8 <= x <= 1.25)}/10 within band"; add(c)

    for cid, name, marks in (("B2", "sine", 4), ("B3", "noise", 2)):
        c = Check(cid, f"{name}: SNR slope in [5.5, 7] dB per bit", marks)
        sl = num(sw.get(name, {}).get("slope_db_per_bit"))
        c.ok = sl is not None and 5.5 <= sl <= 7; c.earned = c.marks if c.ok else 0.0
        c.detail = f"slope {sl}"; add(c)

    c = Check("B4", "Sine SNR > noise SNR at every width from 3 to 8 bits", 3)
    oks = []
    for b in range(3, 9):
        a, n = num(rows["sine"].get(b, {}).get("snr")), num(rows["noise"].get(b, {}).get("snr"))
        oks.append(all_num(a, n) and a > n)
    frac(c, oks); c.detail = f"{sum(oks)}/6 widths"; add(c)

    c = Check("B5", "Outlier drop within 3 dB of 20 log10(range ratio)", 3)
    drop, rr = num(ol.get("drop_db")), num(ol.get("range_ratio"))
    if all_num(drop, rr) and rr > 0:
        pred = 20*math.log10(rr); c.ok = abs(drop - pred) <= 3; c.earned = c.marks if c.ok else 0.0
        c.detail = f"drop {drop:.1f} dB vs predicted {pred:.1f} dB"
    else: c.detail = "missing outlier numbers"
    add(c)

    c = Check("B6", "Blocks of 64 recover at least 10 dB of the drop", 3)
    sb, swh = num(ol.get("snr_blocks")), num(ol.get("snr_whole"))
    if all_num(sb, swh):
        c.ok = sb >= swh + 10; c.earned = c.marks if c.ok else 0.0; c.detail = f"blocks {sb:.1f} dB vs one (S, Z) {swh:.1f} dB"
    else: c.detail = "missing outlier SNRs"
    add(c)

    # --- Part 2: a small model ---
    fa, ska = num(m.get("float_acc")), num(m.get("sklearn_acc"))
    acc = {g: {b: num(m.get(g, {}).get(str(b), {}).get("acc")) for b in range(1, 9)} for g in ("tensor", "channel")}
    snr = {g: {b: num(m.get(g, {}).get(str(b), {}).get("snr_w1")) for b in range(1, 9)} for g in ("tensor", "channel")}

    c = Check("C1", "Forward pass reproduces sklearn's accuracy", 2)
    c.ok = all_num(fa, ska) and abs(fa - ska) < 1e-9; c.earned = c.marks if c.ok else 0.0
    c.detail = f"float {fa}, sklearn {ska}"; add(c)

    c = Check("C2", "8-bit accuracy within 0.01 of float, both granularities", 4)
    frac(c, [all_num(fa, acc[g][8]) and acc[g][8] >= fa - 0.01 for g in ("tensor", "channel")])
    c.detail = f"float {fa}, tensor {acc['tensor'][8]}, channel {acc['channel'][8]}"; add(c)

    c = Check("C3", "Collapse: 1-bit near chance, 2-bit per-tensor at least 0.02 below float", 3)
    a1, a2 = acc["tensor"][1], acc["tensor"][2]
    frac(c, [a1 is not None and a1 < 0.3, all_num(a2, fa) and a2 < fa - 0.02])
    c.detail = f"1-bit {a1}, 2-bit {a2}, float {fa}"; add(c)

    c = Check("C4", "Per-channel scales sit on output columns (<= 4 levels per column at 2 bits); acc >= per-tensor", 3)
    t2, c2 = acc["tensor"][2], acc["channel"][2]
    lv = num(m.get("channel", {}).get("2", {}).get("max_levels_per_column"))
    frac(c, [lv is not None and lv <= 4, all_num(t2, c2) and c2 >= t2])
    c.detail = f"max levels in a column {lv}; acc tensor {t2}, channel {c2}"; add(c)

    c = Check("C5", "Per-channel W1 SNR > per-tensor at every width from 2 to 8 bits", 3)
    oks = [all_num(snr["tensor"][b], snr["channel"][b]) and snr["channel"][b] > snr["tensor"][b] for b in range(2, 9)]
    frac(c, oks); c.detail = f"{sum(oks)}/7 widths"; add(c)

    c = Check("C6", "Accuracy within 0.02 of float at 4-8 bits, both granularities", 2)
    oks = [all_num(acc[g][b], fa) and acc[g][b] >= fa - 0.02 for g in ("tensor", "channel") for b in range(4, 9)]
    frac(c, oks); c.detail = f"{sum(oks)}/10"; add(c)

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
    vals = [r.get("snr") for name in ("sine", "noise") for r in R.get("sweep", {}).get(name, {}).get("rows", [])]
    vals += [R.get("outlier", {}).get("snr_whole"), R.get("model", {}).get("float_acc")]
    vals += [R.get("model", {}).get("tensor", {}).get(str(b), {}).get("snr_w1") for b in range(1, 9)]
    return tuple(vals)

def scorecard(s: Submission) -> str:
    env = s.data.get("env", {})
    L = [f"{'='*60}", f"Lab 19-22b  --  {s.roll}  ({s.data.get('name','?')})", "="*60,
         f"Seed: {env.get('seed','?')}   sine cycles: {env.get('cycles','?')}", "",
         f"AUTOMATIC  {s.auto:.1f} / {AUTO_TOTAL}", "-"*60]
    for c in s.checks:
        L.append(f"  [{'x' if c.ok else ' '}] {c.id} {c.desc:<62} {c.earned:4.1f}/{c.marks}")
        if c.detail: L.append(f"        {c.detail}")
    f, t = prose_completeness(s.prose)
    L += ["", f"RUBRIC (human)  -- / {RUBRIC_TOTAL}", "-"*60,
          f"  {f}/{t} 📝 cells filled" + (f"   ({t-f} left as template)" if t and f < t else ""),
          "  Heaviest: 1b's dB-per-bit explanation, 1c's outlier-in-bits paragraph and",
          "  2b's 'why a model tolerates error a signal metric calls large'.", ""]
    for k, v in s.prose.items():
        L.append(f"  --- {k} ---")
        L.append("      " + (v.replace("\n", "\n      ") if v else "(EMPTY / template)"))
    if s.flags: L += ["", "FLAGS: " + "; ".join(s.flags)]
    return "\n".join(L)

def load(directory: Path):
    subs = []
    for jf in sorted(directory.glob(f"submission_{LAB}_*.json")):
        try: data = json.loads(jf.read_text())
        except Exception as e:
            print(f"  !! {jf.name}: bad JSON ({e})", file=sys.stderr); continue
        roll = str(data.get("roll") or jf.stem.replace(f"submission_{LAB}_", ""))
        nb = next(iter(directory.glob(f"*{roll}*.ipynb")), None)
        subs.append(Submission(roll=roll, data=data, nb_path=nb))
    return subs

def main() -> int:
    ap = argparse.ArgumentParser(description="Mark Lab 19-22b submissions.")
    ap.add_argument("directory", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--csv-only", action="store_true")
    a = ap.parse_args()
    subs = load(a.directory)
    if not subs:
        print(f"No submission_{LAB}_*.json in {a.directory}", file=sys.stderr); return 1
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
