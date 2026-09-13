#!/usr/bin/env python3
"""
grade_lab11_12.py -- automated marking for Lab 11/12, "Three formats, one model".

    python grade_lab11_12.py <submissions_dir> [-o report_dir] [--blank Lab11_12.ipynb]

WHAT THIS CAN AND CANNOT MARK
-----------------------------
Same principle as the other DS635 labs: every number is machine-dependent. A Colab
CPU, a laptop SSD and a workstation NVMe disagree on every timing and size, and all
are correct. There is no answer key.

What IS gradable is that the lab validates a model of serialization whose predictions
are *ordinal* and hold on any machine:

    warm load faster than cold           the page cache served the second read
    safetensors RSS delta ~ 0            mmap maps; it does not copy
    safetensors load rchar ~ 0,          mmap issues no read() syscall; an explicit
      explicit read rchar ~ file           read of the same bytes moves the whole file
    pt RSS delta ~ file                  pickle materialises the object graph
    contiguous cost grows with size      packing a view is a real copy
    weights_only blocks, unsafe fires    the mitigation is an allow-list over a
                                           format that can still execute

This marks those relationships between a student's OWN numbers (40) and extracts
their prose for rubric marking (60). It never compares magnitudes to a reference.

DELIBERATELY NOT AUTO-CHECKED
-----------------------------
- The cold/warm *ratio*. On a warm-cached CI box or a RAM-disk it can be ~1.0; a
  student who reports that and explains it is right. A3 only checks warm <= cold
  within noise, not by how much.
- Part 4 (real GGUF). It needs a file the student supplies; its correctness is
  rubric-marked from the prose, not here.
- macOS submissions have null read_bytes/rchar (no /proc). Those checks are skipped,
  not failed -- see A2's platform note.
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
    d = s.data; F = d.get("results", {}).get("formats", {})
    S = d.get("results", {}).get("stride", {}); T = d.get("results", {}).get("trust", {})
    env = d.get("env", {}); add = s.checks.append
    pt, st = F.get("pt", {}), F.get("safetensors", {})
    linux = bool(env.get("linux"))

    c = Check("A1", "Provenance: roll, name, torch, safetensors", 3)
    have = [bool(s.roll), bool(d.get("name")), bool(env.get("torch")), bool(env.get("safetensors"))]
    c.earned = round(c.marks*sum(have)/len(have), 2); c.ok = all(have); c.detail = f"{sum(have)}/4"; add(c)

    c = Check("A2", "Part 1 records .pt and .safetensors with load metrics", 4)
    need = ["file_mb", "cold_s", "warm_s", "load_rss_delta_mb"]
    have = sum(1 for blk in (pt, st) for k in need if num(blk.get(k)) is not None)
    c.earned = round(c.marks*have/(2*len(need)), 2); c.ok = have == 2*len(need); c.detail = f"{have}/{2*len(need)} metrics"; add(c)

    c = Check("A3", "Warm load <= cold load (both formats, noise slack)", 6)
    oks = []
    for name, blk in (("pt", pt), ("safetensors", st)):
        cold, warm = num(blk.get("cold_s")), num(blk.get("warm_s"))
        oks.append(cold is not None and warm is not None and warm <= cold*1.15)
    c.earned = round(c.marks*sum(oks)/2, 2); c.ok = all(oks); c.detail = f"{sum(oks)}/2 formats"; add(c)

    c = Check("A4", "safetensors load RSS delta << file (mmap, not copy)", 6)
    rss, fmb = num(st.get("load_rss_delta_mb")), num(st.get("file_mb"))
    if rss is not None and fmb:
        c.ok = rss < 0.25*fmb; c.earned = c.marks if c.ok else 0.0
        c.detail = f"+{rss:.1f} MB vs {fmb:.0f} MB file"
    else: c.detail = "missing"
    add(c)

    c = Check("A5", "pt load RSS delta ~ file (pickle materialises)", 5)
    rss, fmb = num(pt.get("load_rss_delta_mb")), num(pt.get("file_mb"))
    if rss is not None and fmb:
        c.ok = rss >= 0.5*fmb; c.earned = c.marks if c.ok else 0.0
        c.detail = f"+{rss:.1f} MB vs {fmb:.0f} MB file"
    else: c.detail = "missing"
    add(c)

    c = Check("A6", "safetensors rchar ~ 0 on load, ~ file on explicit read", 6)
    if not linux:
        c.earned = c.marks; c.ok = True; c.detail = "non-Linux: no /proc, skipped in favour"
    else:
        lr, rr, fmb = num(st.get("load_rchar_mb")), num(st.get("explicit_read_rchar_mb")), num(st.get("file_mb"))
        if None not in (lr, rr, fmb) and fmb:
            c.ok = lr < 0.25*fmb and rr >= 0.5*fmb
            c.earned = c.marks if c.ok else round(c.marks*(0.5*(lr < 0.25*fmb)+0.5*(rr >= 0.5*fmb)), 2)
            c.detail = f"load rchar {lr:.1f}, read rchar {rr:.1f}, file {fmb:.0f}"
        else: c.detail = "missing rchar data"
    add(c)

    c = Check("A7", "Transpose shares buffer AND safetensors refuses the view", 4)
    a, b = S.get("transpose_same_ptr"), S.get("safetensors_refused_view")
    c.ok = a is True and b is True; c.earned = c.marks*(bool(a is True)+bool(b is True))/2
    c.detail = f"same_ptr={a}, refused={b}"; add(c)

    c = Check("A8", "Contiguous cost grows with tensor size", 3)
    sz = [num(x) for x in (S.get("contig_sizes_mb") or [])]
    tm = [num(x) for x in (S.get("contig_times_ms") or [])]
    if len(tm) >= 3 and all(v is not None for v in tm):
        drops = sum(1 for x, y in zip(tm, tm[1:]) if y < 0.9*x)
        c.ok = drops == 0 and tm[-1] > tm[0]; c.earned = c.marks if c.ok else 0.0
        c.detail = f"{len(tm)} points, {drops} inversion(s), {tm[0]:.1f}->{tm[-1]:.1f} ms"
    else: c.detail = "fewer than 3 timing points"
    add(c)

    c = Check("A9", "Payload blocked by default, fires only under weights_only=False", 3)
    fd, bl, fu = T.get("payload_fired_default"), T.get("blocked_by_weights_only"), T.get("payload_fired_unsafe")
    c.ok = (fd is False and bl is True and fu is True); c.earned = c.marks if c.ok else 0.0
    c.detail = f"default_fired={fd}, blocked={bl}, unsafe_fired={fu}"; add(c)

PLACEHOLDER = "your "  # template cells contain "(your ... here)"
def extract_prose(nb_path: Path) -> dict:
    try: nb = json.loads(nb_path.read_text())
    except Exception: return {}
    out = {}; i = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown": continue
        txt = "".join(cell.get("source", []))
        if "📝" in txt:
            i += 1
            # student prose = any line after the prompt that is not the italic placeholder
            body = [l for l in txt.splitlines() if l.strip() and not l.strip().startswith("📝")
                    and not (l.strip().startswith("*(") and l.strip().endswith(")*"))
                    and PLACEHOLDER not in l]
            out[f"prompt_{i}"] = "\n".join(body).strip()
    return out

def prose_completeness(prose: dict):
    if not prose: return 0, 0
    filled = sum(1 for v in prose.values() if len(v) > 20)
    return filled, len(prose)

def fingerprint(s: Submission):
    F = s.data.get("results", {}).get("formats", {})
    vals = []
    for blk in F.values():
        for k in ("cold_s", "warm_s", "load_rss_delta_mb"): vals.append(blk.get(k))
    vals += (s.data.get("results", {}).get("stride", {}).get("contig_times_ms") or [])
    return tuple(vals)

def scorecard(s: Submission) -> str:
    L = [f"{'='*60}", f"Lab 11/12  --  {s.roll}  ({s.data.get('name','?')})", "="*60,
         f"Platform: {s.data.get('env',{}).get('platform','?')}", "",
         f"AUTOMATIC  {s.auto:.1f} / {AUTO_TOTAL}", "-"*60]
    for c in s.checks:
        L.append(f"  [{'x' if c.ok else ' '}] {c.id} {c.desc:<48} {c.earned:4.1f}/{c.marks}")
        if c.detail: L.append(f"        {c.detail}")
    f, t = prose_completeness(s.prose)
    L += ["", f"RUBRIC (human)  -- / {RUBRIC_TOTAL}", "-"*60,
          f"  {f}/{t} 📝 cells filled" + (f"   ({t-f} left as template)" if t and f < t else ""),
          "  Rubric weighting: Part 1 explanation, Part 2 'who pays', and Part 3's",
          "  'what safetensors cannot protect' paragraph carry the most marks.", ""]
    for k, v in s.prose.items():
        L.append(f"  --- {k} ---")
        L.append("      " + (v.replace("\n", "\n      ") if v else "(EMPTY / template)"))
    if s.flags: L += ["", "FLAGS: " + "; ".join(s.flags)]
    return "\n".join(L)

def load(directory: Path):
    subs = []
    for jf in sorted(directory.glob("submission_lab11_12_*.json")):
        try: data = json.loads(jf.read_text())
        except Exception as e:
            print(f"  !! {jf.name}: bad JSON ({e})", file=sys.stderr); continue
        roll = str(data.get("roll") or jf.stem.replace("submission_lab11_12_", ""))
        nb = next(iter(directory.glob(f"*{roll}*.ipynb")), None)
        subs.append(Submission(roll=roll, data=data, nb_path=nb))
    return subs

def main() -> int:
    ap = argparse.ArgumentParser(description="Mark Lab 11/12 submissions.")
    ap.add_argument("directory", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--csv-only", action="store_true")
    a = ap.parse_args()
    subs = load(a.directory)
    if not subs:
        print(f"No submission_lab11_12_*.json in {a.directory}", file=sys.stderr); return 1

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
