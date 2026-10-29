#!/usr/bin/env python3
"""
grade_lab19_22a.py -- automated marking for Lab 19-22a, "Fixed point and affine quantization, by hand".

    python grade_lab19_22a.py <submissions_dir> --tests <hidden_tests_dir> [-o report_dir] [--csv-only]

<submissions_dir> holds one folder per student, named by roll number, each with the three submitted
files: fixed_point.py, affine_quantizer.py, answers.md. <hidden_tests_dir> holds
test_hidden_fixed_point.py and test_hidden_affine.py, which are kept out of the course repository.

Each student file is copied into a fresh temporary directory next to its hidden test and run under
pytest in a subprocess with a timeout. This executes student code with the marker's permissions:
run the script inside a throwaway container or VM.

Marks are per TODO: each TODO's weight times the fraction of its hidden tests that pass. A test's
TODO is the prefix of its name (test_fmul_... -> fmul).

    FixedPoint        fa scale 3 · fb range 4 · fenc encode 4 · fc saturate 4 · fadd add 4 · fmul D-F 9   = 28
    AffineQuantizer   aa grid 4 · ab stretch 5 · ac scale 3 · ad zero-point 6 · ae quantize 8 · af dequantize 6 = 32
    TODO G (stretch)  ag four-term dot product, up to 4 bonus marks on top of the 60

60 automatic + 40 rubric (answers.md, extracted for a human). Identical answers.md files are flagged.
"""
import argparse, csv, hashlib, re, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

AUTO_TOTAL, RUBRIC_TOTAL, BONUS_TOTAL = 60, 40, 4
TIMEOUT_S = 120

SUITES = {
    "fixed_point.py": ("test_hidden_fixed_point.py", {
        "fa": ("TODO A  scale", 3), "fb": ("TODO B  range", 4), "fenc": ("encode", 4),
        "fc": ("TODO C  saturate", 4), "fadd": ("addition", 4), "fmul": ("TODO D-F  multiply", 9)}),
    "affine_quantizer.py": ("test_hidden_affine.py", {
        "aa": ("TODO A  grid", 4), "ab": ("TODO B  stretch to zero", 5), "ac": ("TODO C  scale", 3),
        "ad": ("TODO D  zero-point", 6), "ae": ("TODO E  quantize", 8), "af": ("TODO F  dequantize", 6),
        "ag": ("TODO G  dot product (bonus)", BONUS_TOTAL)}),
}
BONUS = {"ag"}

@dataclass
class Item:
    file: str; todo: str; desc: str; marks: float
    passed: int = 0; total: int = 0
    @property
    def earned(self): return round(self.marks * self.passed / self.total, 2) if self.total else 0.0

@dataclass
class Submission:
    roll: str; folder: Path
    items: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    answers: dict = field(default_factory=dict)
    name: str = ""
    @property
    def auto(self): return round(sum(i.earned for i in self.items if i.todo not in BONUS), 2)
    @property
    def bonus(self): return round(sum(i.earned for i in self.items if i.todo in BONUS), 2)

def run_suite(student_file: Path, test_file: Path):
    """Return {test_name: passed} for one student file, or None when the suite could not run."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(student_file, tmp / student_file.name)
        shutil.copy(test_file, tmp / test_file.name)
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               f"--junitxml={tmp / 'r.xml'}", test_file.name]
        try:
            subprocess.run(cmd, cwd=tmp, capture_output=True, timeout=TIMEOUT_S,
                           env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
                                "PYTHONPATH": ":".join(p for p in sys.path if p)})
        except subprocess.TimeoutExpired:
            return None
        if not (tmp / "r.xml").exists(): return None
        out = {}
        for tc in ET.parse(tmp / "r.xml").getroot().iter("testcase"):
            failed = any(child.tag in ("failure", "error") for child in tc)
            skipped = any(child.tag == "skipped" for child in tc)
            out[tc.get("name")] = not (failed or skipped)
        return out

def mark(s: Submission, tests_dir: Path):
    for fname, (tname, todos) in SUITES.items():
        f = s.folder / fname
        results = run_suite(f, tests_dir / tname) if f.exists() else None
        if not f.exists(): s.flags.append(f"{fname} missing")
        elif results is None: s.flags.append(f"{fname}: suite did not run (timeout or import error)")
        collected = 0
        for todo, (desc, marks) in todos.items():
            it = Item(fname, todo, desc, marks)
            if results:
                mine = [ok for name, ok in results.items() if name.startswith(f"test_{todo}_")]
                it.passed, it.total = sum(mine), len(mine)
            collected += it.total
            s.items.append(it)
        if results is not None and collected == 0:
            s.flags.append(f"{fname}: no tests collected (syntax or import error)")

PLACEHOLDER = re.compile(r"^\*\(your answer.*\)\*$", re.I)
def read_answers(s: Submission):
    f = s.folder / "answers.md"
    if not f.exists(): s.flags.append("answers.md missing"); return
    txt = f.read_text(errors="replace")
    m = re.search(r"^\s*\**Name\**\s*:\s*(.+)$", txt, re.M | re.I)
    s.name = m.group(1).strip() if m else ""
    for block in re.split(r"^## ", txt, flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        lines = [l for l in body.splitlines() if l.strip() and not l.lstrip().startswith(">")
                 and not PLACEHOLDER.match(l.strip())]
        s.answers[head.strip()] = "\n".join(lines).strip()

def answers_filled(s: Submission):
    return sum(1 for v in s.answers.values() if len(v) > 20), len(s.answers)

def text_hash(path: Path):
    """Hash of the answers with the Roll and Name lines and whitespace removed. Correct code is often
    byte-identical between honest students, since the TODOs are one-liners in a shared template;
    identical prose is the copying signal."""
    lines = [l.strip() for l in path.read_text(errors="replace").splitlines()
             if l.strip() and not re.match(r"^\**(roll|name)\**\s*:", l.strip(), re.I)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()

def scorecard(s: Submission) -> str:
    L = ["="*64, f"Lab 19-22a  --  {s.roll}  ({s.name or '?'})", "="*64, "",
         f"AUTOMATIC  {s.auto:.1f} / {AUTO_TOTAL}    bonus {s.bonus:.1f} / {BONUS_TOTAL}", "-"*64]
    current = None
    for it in s.items:
        if it.file != current:
            current = it.file; L.append(f"  {it.file}")
        ok = it.total and it.passed == it.total
        L.append(f"    [{'x' if ok else ' '}] {it.desc:<30} {it.passed:3d}/{it.total:<3d} tests   {it.earned:4.1f}/{it.marks}")
    f, t = answers_filled(s)
    L += ["", f"RUBRIC (human)  -- / {RUBRIC_TOTAL}", "-"*64,
          f"  {f}/{t} answers filled" + (f"   ({t-f} empty)" if t and f < t else ""), ""]
    for k, v in s.answers.items():
        L.append(f"  --- {k} ---")
        L.append("      " + (v.replace("\n", "\n      ") if v else "(EMPTY)"))
    if s.flags: L += ["", "FLAGS: " + "; ".join(s.flags)]
    return "\n".join(L)

def main() -> int:
    ap = argparse.ArgumentParser(description="Mark Lab 19-22a submissions.")
    ap.add_argument("directory", type=Path, help="one folder per roll number")
    ap.add_argument("--tests", type=Path, required=True, help="folder with the hidden test files")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--csv-only", action="store_true")
    a = ap.parse_args()
    for tname, _ in SUITES.values():
        if not (a.tests / tname).exists():
            print(f"missing hidden test file {a.tests / tname}", file=sys.stderr); return 1
    subs = [Submission(roll=d.name, folder=d) for d in sorted(a.directory.iterdir()) if d.is_dir()]
    if not subs:
        print(f"No submission folders in {a.directory}", file=sys.stderr); return 1
    seen = {}
    for s in subs:
        mark(s, a.tests); read_answers(s)
        if answers_filled(s)[0] == 0: continue
        h = text_hash(s.folder / "answers.md")
        if h in seen: s.flags.append(f"answers.md identical to {seen[h]}'s")
        else: seen[h] = s.roll
    out = a.out or a.directory
    out.mkdir(exist_ok=True, parents=True)
    if not a.csv_only:
        for s in subs: (out / f"report_{s.roll}.txt").write_text(scorecard(s))
    with open(out / "marks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roll", "name", "auto", "auto_total", "bonus", "answers_filled", "answers_total", "flags"])
        for s in subs:
            fl, tt = answers_filled(s)
            w.writerow([s.roll, s.name, s.auto, AUTO_TOTAL, s.bonus, fl, tt, "; ".join(s.flags)])
    print(f"marked {len(subs)} submission(s)")
    for s in subs:
        print(f"  {s.roll:12s} auto {s.auto:5.1f}/{AUTO_TOTAL}  bonus {s.bonus:3.1f}"
              + (f"   FLAG: {'; '.join(s.flags)}" if s.flags else ""))
    print(f"wrote {out / 'marks.csv'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
