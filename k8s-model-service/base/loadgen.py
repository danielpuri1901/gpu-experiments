"""Load generator and result analysis (stdlib only).
  run URL [--seconds S] [--count N] [--workers W] [--rps R]   prints one JSON line
  summary FILE [T0 T1] | timeline FILE BUCKET | share K N P | until FILE T0 OP N | recover FILE T0 N"""
import json, sys, threading, time, urllib.request, urllib.error
from collections import Counter

BODY = json.dumps({"model": "stub-llm", "messages": [{"role": "user", "content": "ping"}]}).encode()


def one(url):
    t = time.time()
    status = 0
    try:
        req = urllib.request.Request(url, BODY, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            status, h = r.status, r.headers
            r.read()
    except urllib.error.HTTPError as e:
        status, h = e.code, e.headers
    except Exception:
        h = {}
    ver, pod = h.get("X-Model-Version", ""), h.get("X-Pod", "")
    return [round(t, 3), status, round((time.time() - t) * 1000, 1), ver, pod]


def run(url, seconds=0, count=0, workers=4, rps=0):
    reqs, lock, sent = [], threading.Lock(), [0]
    stop = time.time() + seconds if seconds else None
    gap = workers / rps if rps else 0

    def worker():
        nxt = time.time()
        while True:
            with lock:
                if (count and sent[0] >= count) or (stop and time.time() >= stop):
                    return
                sent[0] += 1
            r = one(url)
            with lock:
                reqs.append(r)
            nxt += gap
            time.sleep(max(0, nxt - time.time()))

    ts = [threading.Thread(target=worker) for _ in range(workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    reqs.sort()
    print(json.dumps({"url": url, "start": reqs[0][0] if reqs else 0, "requests": reqs}), flush=True)
    try:  # let a sidecar exit so the Job can finish
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:15020/quitquitquit", b"", method="POST"), timeout=2)
    except Exception:
        pass


def window(reqs, t0=None, t1=None):
    return [r for r in reqs if (t0 is None or r[0] >= t0) and (t1 is None or r[0] < t1)]


def summarize(reqs, t0=None, t1=None):
    r = window(reqs, t0, t1)
    ok = [x for x in r if x[1] == 200]
    lat = sorted(x[2] for x in ok)
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None
    return {"n": len(r), "failed": len(r) - len(ok), "p50_ms": pct(0.5), "p95_ms": pct(0.95),
            "versions": dict(Counter(x[3] for x in ok)), "pods": len({x[4] for x in ok})}


def timeline(reqs, bucket):
    """Rows of (second offset, {version: count}, failed) per time bucket."""
    if not reqs:
        return []
    t0, rows = reqs[0][0], {}
    for ts, status, _, ver, _ in reqs:
        row = rows.setdefault(int((ts - t0) // bucket) * bucket, [Counter(), 0])
        if status == 200:
            row[0][ver] += 1
        else:
            row[1] += 1
    return [(k, dict(v), f) for k, (v, f) in sorted(rows.items())]


def share_within(k, n, p, z=4):
    """Is k/n within z standard errors of the target share p?"""
    tol = z * (p * (1 - p) / n) ** 0.5
    return abs(k / n - p) <= tol + 1e-9, k / n, tol


def time_until(trace, t0, pred):
    """Seconds after t0 of the first (time, value) sample that satisfies pred, or None."""
    return next((t - t0 for t, v in trace if t >= t0 and pred(v)), None)


def recover_seconds(trace, t0, target):
    """Seconds from t0 until value is back at target after a dip below it (0 if no dip)."""
    dip = time_until(trace, t0, lambda v: v < target)
    if dip is None:
        return 0.0
    back = time_until(trace, t0 + dip, lambda v: v >= target)
    return None if back is None else dip + back


def load_trace(path):
    return [(float(a), int(b)) for a, b in (ln.split() for ln in open(path) if len(ln.split()) == 2)]


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        o = dict(zip(a[1::2], map(float, a[2::2])))
        run(a[0], o.get("--seconds", 0), int(o.get("--count", 0)), int(o.get("--workers", 4)), o.get("--rps", 0))
    elif cmd == "summary":
        print(json.dumps(summarize(json.load(open(a[0]))["requests"], *map(float, a[1:]))))
    elif cmd == "timeline":
        for s, v, f in timeline(json.load(open(a[0]))["requests"], float(a[1])):
            print(f"t+{s:>3.0f}s " + " ".join(f"{k}={n}" for k, n in sorted(v.items())) + f" failed={f}")
    elif cmd == "share":
        ok, share, tol = share_within(int(a[0]), int(a[1]), float(a[2]))
        print(json.dumps({"ok": ok, "share": round(share, 3), "tol": round(tol, 3)}))
    elif cmd == "until":
        ops = {"gt": lambda x: x > int(a[3]), "le": lambda x: x <= int(a[3]), "ge": lambda x: x >= int(a[3])}
        print(-1 if (r := time_until(load_trace(a[0]), float(a[1]), ops[a[2]])) is None else round(r, 1))
    elif cmd == "recover":
        print(-1 if (r := recover_seconds(load_trace(a[0]), float(a[1]), int(a[2]))) is None else round(r, 1))
