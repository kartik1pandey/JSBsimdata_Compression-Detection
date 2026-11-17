"""Flight viewer: a small local web server (Python standard library + pandas).

    python viewer/server.py                 # then open http://localhost:8050
    python viewer/server.py --data D:/path/to/data --port 8050

API (JSON):
    /api/manifest                 flights + anomaly catalogue text
    /api/channels                 every logged channel with subsystem / unit / meaning
    /api/flight/<id>              scenario, anomaly, per-channel effect scores
    /api/series/<id>?ch=a,b       time series of channels a, b (+ the normal twin when there is one)
                                  'truth:sensor/ALT' = the TRUE value behind a sensor reading,
                                  converted to the sensor's unit
"""
import argparse
import json
import math
import sys
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.anomalies import CATALOG, SENSED  # noqa: E402

STATIC = Path(__file__).resolve().parent / 'static'
DOCS = ROOT / 'docs'

# sensor reading -> (true JSBSim property, factor into the sensor's unit)
TRUTH = {f'sensor/{k}': (v[0], 60.0 if k == 'VS' else (180.0 / math.pi if v[0].endswith('rad_sec') else 1.0))
         for k, v in SENSED.items()}
TRUTH.update({f'sensor/TLA{e + 1}': (f'fcs/throttle-cmd-norm[{e}]', 1.0) for e in range(4)})


class Store:
    def __init__(self, data):
        self.data = Path(data)
        self.cache = OrderedDict()

    def json(self, name):
        return json.loads((self.data / name).read_text(encoding='utf-8'))

    def frame(self, key):
        if key not in self.cache:
            p = self.data / 'flights' / f'{key}.parquet'
            if not p.exists():
                return None
            self.cache[key] = pd.read_parquet(p)
            while len(self.cache) > 6:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def _col(df, name):
    if name.startswith('truth:'):
        src = name[6:]
        prop, k = TRUTH[src]
        v = df[prop].to_numpy(float) * k
        return np.mod(v, 360.0) if src == 'sensor/HDG' else v
    return df[name].to_numpy(float)


def _clean(v):
    v = np.round(v.astype(float), 6)
    return [None if not math.isfinite(x) else x for x in v.tolist()]


class Handler(SimpleHTTPRequestHandler):
    store: Store = None

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, allow_nan=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        try:
            if path == '/api/manifest':
                cat = {k: dict(group=c['group'], formula=c['formula'], text=c['text'],
                               profiles=c['profiles'], variables=list(c['vars']),
                               bounds=c['bounds'], max_dur=c['max_dur']) for k, c in CATALOG.items()}
                return self._send(dict(flights=self.store.json('manifest.json'), catalog=cat))
            if path == '/api/channels':
                return self._send(self.store.json('channels.json'))
            if path.startswith('/api/flight/'):
                return self._send(self.store.json(f"meta/{path.split('/')[-1]}.json"))
            if path.startswith('/api/series/'):
                fid = path.split('/')[-1]
                chans = [c for c in parse_qs(u.query).get('ch', [''])[0].split(',') if c]
                df = self.store.frame(fid)
                if df is None:
                    return self._send({'error': f'no flight {fid}'}, 404)
                out = dict(t=_clean(df['t'].to_numpy()), label=df['label'].astype(int).tolist(),
                           series={c: _clean(_col(df, c)) for c in chans}, twin=None)
                tw = self.store.frame(f'{fid}_twin')
                if tw is not None:
                    out['twin'] = dict(t=_clean(tw['t'].to_numpy()),
                                       series={c: _clean(_col(tw, c)) for c in chans})
                return self._send(out)
            if path.startswith('/docs/'):
                f = (DOCS / path[len('/docs/'):]).resolve()
                if f.is_file() and DOCS.resolve() in f.parents:
                    body = f.read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/markdown; charset=utf-8'
                                     if f.suffix == '.md' else 'application/octet-stream')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                return self._send({'error': 'not found'}, 404)
        except FileNotFoundError as e:
            return self._send({'error': str(e)}, 404)
        except KeyError as e:
            return self._send({'error': f'unknown channel {e}'}, 400)
        if path in ('/', ''):
            self.path = '/index.html'
        return super().do_GET()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data'))
    ap.add_argument('--port', type=int, default=8050)
    args = ap.parse_args()
    if not (Path(args.data) / 'manifest.json').exists():
        raise SystemExit(f'no manifest.json in {args.data} -- run  python -m lab.generate  first')
    Handler.store = Store(args.data)
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'viewer on http://localhost:{args.port}  (data: {args.data})  Ctrl+C to stop')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
