"""Stand-in for a NIM: a stdlib HTTP model server with OpenAI-style routes.
It burns CPU per request instead of running a model, so no GPU is needed."""
import json, os, signal, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = os.environ.get("VERSION", "v1")
POD = os.environ.get("POD_NAME", socket.gethostname())
LOAD_SECONDS = float(os.environ.get("LOAD_SECONDS", "5"))  # simulated model load
BURN_MS = float(os.environ.get("BURN_MS", "20"))           # CPU burned per request
NEVER_READY = os.environ.get("NEVER_READY") == "1"         # a bad release
START = time.time()
state = {"broken": False, "poisoned": False, "draining": False, "inflight": 0, "total": 0, "failed": 0}
lock = threading.Lock()


def loaded():
    return time.time() - START >= LOAD_SECONDS


def ready():
    return loaded() and not (NEVER_READY or state["broken"] or state["draining"])


def burn(ms):
    end = time.thread_time() + ms / 1000
    while time.thread_time() < end:
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Model-Version", VERSION)
        self.send_header("X-Pod", POD)
        self.send_header("Connection", "close")  # no idle keep-alive, so drain ends fast
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":  # liveness: is the process alive and sane?
            self.reply(500 if state["broken"] else 200, {"status": "broken" if state["broken"] else "ok"})
        elif p == "/ready":  # readiness: should the Service send traffic here?
            self.reply(200 if ready() else 503, {"ready": ready()})
        elif p == "/break":  # makes /health fail, so liveness restarts the pod
            state["broken"] = True
            self.reply(200, {"broken": True})
        elif p == "/poison":  # makes completions return 500 while probes stay green
            state["poisoned"] = True
            self.reply(200, {"poisoned": True})
        elif p == "/v1/models":
            self.reply(200, {"object": "list", "data": [{"id": "stub-llm", "object": "model", "owned_by": "demo"}]})
        elif p == "/metrics":
            lines = [("model_requests_total", state["total"]), ("model_requests_failed_total", state["failed"]),
                     ("model_inflight_requests", state["inflight"]), ("model_loaded", int(loaded())),
                     ("process_cpu_seconds_total", round(time.process_time(), 4))]
            self.reply(200, "".join(f"{k} {v}\n" for k, v in lines), "text/plain")
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/chat/completions":
            return self.reply(404, {"error": "not found"})
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with lock:
            state["inflight"] += 1
            state["total"] += 1
        try:
            if not loaded():
                code, body = 503, {"error": "model loading"}
            elif state["poisoned"]:
                code, body = 500, {"error": "poisoned"}
            else:
                burn(BURN_MS)  # imitates inference
                code, body = 200, {"id": "chatcmpl-demo", "object": "chat.completion", "created": int(time.time()),
                                   "model": f"stub-llm-{VERSION}", "version": VERSION, "pod": POD,
                                   "choices": [{"index": 0, "finish_reason": "stop",
                                                "message": {"role": "assistant", "content": "pong"}}]}
            if code != 200:
                state["failed"] += 1
            self.reply(code, body)
        finally:
            with lock:
                state["inflight"] -= 1


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), Handler)
    srv.daemon_threads = False  # server_close() waits for in-flight requests

    def drain(*_):  # graceful drain: stop accepting, finish in-flight, exit 0
        state["draining"] = True
        threading.Thread(target=srv.shutdown).start()

    signal.signal(signal.SIGTERM, drain)
    srv.serve_forever()
    srv.server_close()


if __name__ == "__main__":
    main()
