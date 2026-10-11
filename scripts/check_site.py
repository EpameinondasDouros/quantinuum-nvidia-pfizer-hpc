"""Check every href/src/poster in site/*.html.

Local links must be relative (the site is served under a sub-path, https://udsy19.github.io/genqai/),
must resolve to a file in site/, and any #fragment must be an id on the target page. With --base URL,
every local target is also fetched over HTTP and must return 200. With --external, http(s) links are
fetched too (non-fatal: some hosts refuse scripted requests).

    .venv/bin/python -m scripts.check_site [--base http://localhost:8000/] [--external]
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

SITE = Path(__file__).resolve().parents[1] / "site"
ATTRS = {"href", "src", "poster"}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k == "id" and v:
                self.ids.add(v)
            if k in ATTRS and v and not (tag == "link" and dict(attrs).get("rel") in ("preconnect",)):
                self.links.append((tag, k, v))


def parse(path: Path) -> Links:
    p = Links()
    p.feed(path.read_text())
    return p


def fetch(url: str) -> int:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "genqai-site-check"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="served site root, e.g. http://localhost:8000/")
    ap.add_argument("--external", action="store_true")
    args = ap.parse_args()
    pages = sorted(SITE.glob("*.html"))
    parsed = {p.name: parse(p) for p in pages}
    errors, checked, external = [], set(), set()
    for page, links in parsed.items():
        for tag, attr, v in links.links:
            u = urlsplit(v)
            if u.scheme in ("http", "https"):
                external.add(v)
                continue
            if u.scheme in ("data", "mailto"):
                continue
            if v.startswith("/"):
                errors.append(f"{page}: absolute path {v} breaks under the /genqai/ sub-path")
                continue
            target = u.path or page
            target = "index.html" if target in ("./", ".") else target
            f = SITE / target
            if not f.is_file():
                errors.append(f"{page}: <{tag} {attr}={v}> -> missing {target}")
                continue
            if u.fragment and f.suffix == ".html" and u.fragment not in parsed[f.name].ids:
                errors.append(f"{page}: {v} -> no id '{u.fragment}' in {f.name}")
            checked.add(u.path or page)
    if args.base:
        for t in sorted(checked | {p.name for p in pages}):
            code = fetch(urljoin(args.base, t))
            print(f"{code} {t}")
            if code != 200:
                errors.append(f"HTTP {code} for {t}")
    if args.external:
        for v in sorted(external):
            code = fetch(v)
            print(f"{code} {v}" + ("" if code == 200 else "   (external; check by hand)"))
    print(f"{len(pages)} pages, {len(checked)} local targets, {len(external)} external links")
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
