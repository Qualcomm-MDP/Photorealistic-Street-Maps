"""Stage 1 (user): draw the streets to align on a map.

Opens route_gui.html in the browser, served from a small local web server. Click along the
streets you want (turns are fine; "New street" starts a separate line), set the corridor
width and press "Use this route". The route is saved as a ROI file:

  {"name": ..., "created": ..., "corridor_m": 15,
   "lines": [[[lon, lat], ...], ...],          the drawn lines
   "bbox": [min_lon, min_lat, max_lon, max_lat]}  box around the lines plus the corridor

The Mapillary layer (mapillary_layer.py) takes this file and fetches the cameras inside the
corridor.

Usage:
  python route_gui.py                      # saves routes/<name>_<time>.json
  python route_gui.py --out my_route.json
"""

import argparse
import json
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from common import ROUTES_DIR, LocalFrame, line_length_m

PAGE = Path(__file__).resolve().parent / "route_gui.html"


def make_roi(name, lines, corridor_m):
    """Validate drawn lines and add the search box around them."""
    lines = [[(float(lon), float(lat)) for lon, lat in line] for line in lines if len(line) >= 2]
    if not lines:
        raise ValueError("draw at least one line with two points")
    if not 1 <= corridor_m <= 200:
        raise ValueError("corridor must be 1-200 m")
    pts = [p for line in lines for p in line]
    frame = LocalFrame(*pts[0])
    pad_lon, pad_lat = corridor_m / frame.kx, corridor_m / frame.ky
    lons, lats = [p[0] for p in pts], [p[1] for p in pts]
    return {
        "name": name, "created": datetime.now().isoformat(timespec="seconds"),
        "corridor_m": float(corridor_m), "lines": lines,
        "bbox": [min(lons) - pad_lon, min(lats) - pad_lat, max(lons) + pad_lon, max(lats) + pad_lat],
    }


def select_roi(out=None, port=0, open_browser=True):
    """Serve the route picker until a route is sent; return (roi dict, saved path)."""
    result, done = {}, threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/roi":
                return self.reply(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                roi = make_roi(str(body.get("name") or "route"), body.get("lines", []), float(body.get("corridor_m", 15)))
            except (ValueError, TypeError, KeyError) as e:
                return self.reply(400, {"error": str(e)})
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in roi["name"])
            path = Path(out) if out else ROUTES_DIR / f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(roi, indent=1))
            result.update(roi=roi, path=path)
            self.reply(200, {"saved": str(path)})
            done.set()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Route picker at {url}  (draw the route, then press 'Use this route'; Ctrl+C to cancel)")
    if open_browser:
        webbrowser.open(url)
    try:
        while not done.wait(0.5):
            pass
    except KeyboardInterrupt:
        raise SystemExit("Cancelled; no route saved")
    finally:
        server.shutdown()
    roi = result["roi"]
    length = sum(line_length_m(line) for line in roi["lines"])
    print(f"Route '{roi['name']}': {len(roi['lines'])} line(s), {length:.0f} m, corridor {roi['corridor_m']:.0f} m "
          f"-> {result['path']}")
    return roi, result["path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="Where to save the ROI (default routes/<name>_<time>.json)")
    parser.add_argument("--port", type=int, default=0, help="Local port (default: any free port)")
    parser.add_argument("--no-browser", action="store_true", help="Only print the URL")
    args = parser.parse_args()
    select_roi(args.out, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
