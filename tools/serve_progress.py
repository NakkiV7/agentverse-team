"""Serve runtime/ so the progress page can be watched live: http://localhost:8850/progress.html"""
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent / "runtime"


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


if __name__ == "__main__":
    RUNTIME.mkdir(exist_ok=True)
    shutil.copy(HERE / "progress.html", RUNTIME / "progress.html")
    print("progress page: http://localhost:8850/progress.html")
    ThreadingHTTPServer(("127.0.0.1", 8850), partial(Quiet, directory=str(RUNTIME))).serve_forever()
