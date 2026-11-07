"""Run Buddy:  .venv/bin/python -m buddy  [--no-voice] [--no-mic] [--no-browser]

Opens the demo page at http://localhost:8860. Say (or type) "Buddy, let's play a game?".
Needs the Ollama server running (`ollama serve` or the Ollama app) with gemma3:4b pulled.
"""
import argparse
import webbrowser

from . import config as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-voice", action="store_true", help="no speech output (Buddy's lines only on the page)")
    ap.add_argument("--no-mic", action="store_true", help="no microphone (type on the page instead)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the demo page")
    ap.add_argument("--port", type=int, default=C.UI_PORT)
    args = ap.parse_args()

    from .app import App
    from .ui.server import serve
    app = App(voice=not args.no_voice, mic=not args.no_mic)
    serve(app, args.port)
    url = f"http://localhost:{args.port}"
    app.trace("system", f"demo page: {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
