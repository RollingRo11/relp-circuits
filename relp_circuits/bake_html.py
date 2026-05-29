"""Bundle the Transluce circuit-tracer frontend (HTML/JS/CSS assets) plus
our pre-computed graph data into a single self-contained HTML file.

The frontend uses `util.getFile(path)` for all data fetches (cf.
circuits/frontend/assets/util.js), and that function checks
`window.__datacache` first. By pre-populating `window.__datacache` with
Promise-resolved entries for every JSON path the frontend would request,
all `fetch()`-based loads short-circuit to in-memory data — no server
needed.

The frontend also uses a raw `fetch('/api/neuron_exemplars?...')` call
in init-feature-examples.js. We monkey-patch `window.fetch` at the top
of the bundle to intercept that path and return cached exemplar data
(or `{isDead: true}` if not present).
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_BOOTSTRAP_TEMPLATE = """\
<script>
// ─── relp-circuits standalone-HTML bootstrap ───
// Pre-populate window.__datacache AND install a fetch interceptor that
// returns the inlined data for any cached relative path. Belt-and-braces:
// even if util.getFile() misses its in-memory cache for some reason
// (re-entrancy, cache reset, key-format drift), the fetch fallthrough
// short-circuits to inlined data instead of hitting the network. This
// matters for `file://` opens where CORS blocks every real fetch.
window.isLocalServing = true;

(function () {
  const __DATA = __DATA_PLACEHOLDER__;
  const __EXEMPLARS = __EXEMPLARS_PLACEHOLDER__;

  // Cache by util.getFile's key format: path, path-fileType, path-range,
  // path-range-fileType. We populate the common forms so any default call
  // shape hits.
  window.__datacache = {};
  for (const [k, v] of Object.entries(__DATA)) {
    window.__datacache[k] = Promise.resolve(v);
    window.__datacache[k + "-json"] = Promise.resolve(v);
  }

  // Build a path → JSON-string lookup. Keys are the same paths the
  // frontend's util.getFile uses ("./data/...", "./graph_data/...") plus
  // a few normalized variants ("data/...", "/data/...") to be robust to
  // how the browser resolves relative URLs at file:// origin.
  function __normalizePathVariants(k) {
    const variants = new Set([k]);
    if (k.startsWith("./")) {
      const stripped = k.slice(2);
      variants.add(stripped);
      variants.add("/" + stripped);
    }
    return variants;
  }
  const __JSON_FOR_PATH = {};
  for (const [k, v] of Object.entries(__DATA)) {
    const body = JSON.stringify(v);
    for (const p of __normalizePathVariants(k)) {
      __JSON_FOR_PATH[p] = body;
    }
  }

  function __jsonResponse(body) {
    return new Response(body, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  const __orig_fetch = window.fetch.bind(window);
  window.fetch = function (input, opts) {
    let url = (typeof input === "string") ? input : (input && input.url) || "";

    // 1. Cached relative-path data files.
    if (typeof url === "string") {
      // Strip a "file:///..." prefix that the browser may have already
      // resolved against the page's file:// origin.
      let probe = url;
      const fileMatch = probe.match(/^file:\\/\\/[^/]*(\\/.*)$/);
      if (fileMatch) probe = fileMatch[1];
      if (__JSON_FOR_PATH[probe] !== undefined) {
        return Promise.resolve(__jsonResponse(__JSON_FOR_PATH[probe]));
      }
      // Tail-match: if the cached key is "./data/foo.json", any URL ending
      // with "/data/foo.json" also hits.
      for (const k of Object.keys(__JSON_FOR_PATH)) {
        if (k.startsWith("./") && probe.endsWith(k.slice(1))) {
          return Promise.resolve(__jsonResponse(__JSON_FOR_PATH[k]));
        }
      }
    }

    // 2. /api/neuron_exemplars?layer=X&neuron=Y&sign=Z exemplar UI.
    if (typeof url === "string" && url.includes("/api/neuron_exemplars")) {
      const u = new URL(url, "http://localhost");
      const layer = u.searchParams.get("layer");
      const neuron = u.searchParams.get("neuron");
      const sign = u.searchParams.get("sign") || "+";
      const key = `${layer}_${neuron}_${sign === "-" ? "-" : "+"}`;
      const data = __EXEMPLARS[key] || { isDead: true };
      return Promise.resolve(__jsonResponse(JSON.stringify(data)));
    }

    return __orig_fetch(input, opts);
  };

  // Expose for debugging from devtools.
  window.__relpData = __DATA;
})();
</script>
"""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _inline_styles_and_scripts(html: str, asset_root: Path) -> str:
    """Replace <link rel='stylesheet' href='./X.css'> and <script src='./X.js'>
    with inline <style> and <script> blocks. Anything not local (no leading
    './') is left as-is so external CDN scripts (if any) still load."""

    def replace_link(m: re.Match[str]) -> str:
        href = m.group("href")
        if not href.startswith("./"):
            return m.group(0)
        full = asset_root / href[2:]
        if not full.exists():
            return m.group(0)
        css = _read(full)
        return f"<style>\n{css}\n</style>"

    def replace_script(m: re.Match[str]) -> str:
        src = m.group("src")
        if not src.startswith("./"):
            return m.group(0)
        full = asset_root / src[2:]
        if not full.exists():
            return m.group(0)
        js = _read(full)
        # Avoid </script> inside the JS string by splitting it.
        js = js.replace("</script>", "<\\/script>")
        return f"<script>\n{js}\n</script>"

    html = re.sub(
        r"<link\s+rel=['\"]stylesheet['\"]\s+href=['\"](?P<href>[^'\"]+)['\"]\s*/?>",
        replace_link,
        html,
    )
    html = re.sub(
        r"<script\s+src=['\"](?P<src>[^'\"]+)['\"]\s*></script>",
        replace_script,
        html,
    )
    return html


def bake_folder(
    *,
    asset_root: Path,
    bundle_root: Path,
    out_dir: Path,
    title: str | None = None,
    initial_slug: str | None = None,
) -> dict[str, int]:
    """Emit a *folder* the user can download wholesale and open via file://.

    Layout:
      out_dir/
        index.html        — small (~tens of KB): inlined CSS + frontend JS, no data
        data.js           — `window.__RELP_DATA = {<path>: <json>, ...};`
                            plus `window.__RELP_EXEMPLARS = {...};`

    The HTML loads data.js via `<script src="./data.js">`. Script-tag loads
    from file:// are not subject to the CORS rules that block fetch, so this
    works in every browser without a local server.

    Returns {"html_bytes": N, "data_bytes": M}.
    """
    if not (asset_root / "index.html").exists():
        raise FileNotFoundError(f"missing {asset_root}/index.html")
    html = _read(asset_root / "index.html")

    data_payload: dict[str, object] = {}
    meta_path = bundle_root / "data" / "graph-metadata.json"
    if meta_path.exists():
        data_payload["./data/graph-metadata.json"] = json.loads(_read(meta_path))
    for gp in sorted((bundle_root / "graph_data").glob("*.json")):
        data_payload[f"./graph_data/{gp.stem}.json"] = json.loads(_read(gp))

    exemplars_payload: dict[str, object] = {}
    ex_path = bundle_root / "data" / "neuron_exemplars.json"
    if ex_path.exists():
        exemplars_payload = json.loads(_read(ex_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    data_js = (
        "// Auto-generated by relp_circuits.bake_html.bake_folder.\n"
        "// All graph data + exemplars used by the dashboard.\n"
        f"window.__RELP_DATA = {json.dumps(data_payload)};\n"
        f"window.__RELP_EXEMPLARS = {json.dumps(exemplars_payload)};\n"
    )
    data_path = out_dir / "data.js"
    data_path.write_text(data_js, encoding="utf-8")

    # Inline CSS + JS, but install a bootstrap that reads from the globals set
    # by data.js (which is loaded *before* this bootstrap via <script src>).
    html = _inline_styles_and_scripts(html, asset_root)
    bootstrap = _BOOTSTRAP_FROM_GLOBALS
    data_tag = '<script src="./data.js"></script>\n'

    head_open = re.search(r"<head[^>]*>", html, flags=re.IGNORECASE)
    if head_open:
        idx = head_open.end()
        html = html[:idx] + "\n" + data_tag + bootstrap + "\n" + html[idx:]
    else:
        first_resource = re.search(r"<(link|script)\b", html, flags=re.IGNORECASE)
        prefix_end = first_resource.start() if first_resource else len(html)
        meta_iter = list(re.finditer(r"<meta[^>]*>", html[:prefix_end], flags=re.IGNORECASE))
        if meta_iter:
            idx = meta_iter[-1].end()
        else:
            doctype = re.search(r"<!DOCTYPE[^>]*>", html, flags=re.IGNORECASE)
            idx = doctype.end() if doctype else 0
        html = html[:idx] + "\n" + data_tag + bootstrap + "\n" + html[idx:]

    if title:
        html = re.sub(
            r"<title>[^<]*</title>",
            f"<title>{title}</title>",
            html,
            count=1,
            flags=re.IGNORECASE,
        )

    if initial_slug:
        slug_js = (
            "<script>try{history.replaceState(null,'','?slug="
            f"{initial_slug}');}}catch(e){{}}</script>\n"
        )
        html = html.replace(bootstrap, bootstrap + slug_js, 1)

    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return {
        "html_bytes": index_path.stat().st_size,
        "data_bytes": data_path.stat().st_size,
    }


_BOOTSTRAP_FROM_GLOBALS = """\
<script>
// ─── relp-circuits standalone-folder bootstrap ───
// `data.js` (loaded via <script src> above) has set:
//   window.__RELP_DATA      — {"<path>": <json>, ...}
//   window.__RELP_EXEMPLARS — {"<L>_<N>_<sign>": [...], ...}
// We populate util.getFile's cache from those globals and also install a
// fetch interceptor that short-circuits any relative-path fetch the
// frontend issues — `file://` opens block real cross-origin fetches, so
// this guarantees zero network traffic.
window.isLocalServing = true;

(function () {
  const __DATA = window.__RELP_DATA || {};
  const __EXEMPLARS = window.__RELP_EXEMPLARS || {};

  window.__datacache = {};
  for (const [k, v] of Object.entries(__DATA)) {
    window.__datacache[k] = Promise.resolve(v);
    window.__datacache[k + "-json"] = Promise.resolve(v);
  }

  function pathVariants(k) {
    const out = new Set([k]);
    if (k.startsWith("./")) {
      const s = k.slice(2);
      out.add(s);
      out.add("/" + s);
    }
    return out;
  }
  const JSON_FOR_PATH = {};
  for (const [k, v] of Object.entries(__DATA)) {
    const body = JSON.stringify(v);
    for (const p of pathVariants(k)) JSON_FOR_PATH[p] = body;
  }

  function jsonResponse(body) {
    return new Response(body, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  const origFetch = window.fetch ? window.fetch.bind(window) : null;
  window.fetch = function (input, opts) {
    let url = (typeof input === "string") ? input : (input && input.url) || "";
    if (typeof url === "string") {
      let probe = url;
      const m = probe.match(/^file:\\/\\/[^/]*(\\/.*)$/);
      if (m) probe = m[1];
      if (JSON_FOR_PATH[probe] !== undefined) {
        return Promise.resolve(jsonResponse(JSON_FOR_PATH[probe]));
      }
      for (const k of Object.keys(JSON_FOR_PATH)) {
        if (k.startsWith("./") && probe.endsWith(k.slice(1))) {
          return Promise.resolve(jsonResponse(JSON_FOR_PATH[k]));
        }
      }
      if (url.includes("/api/neuron_exemplars")) {
        const u = new URL(url, "http://localhost");
        const layer = u.searchParams.get("layer");
        const neuron = u.searchParams.get("neuron");
        const sign = u.searchParams.get("sign") || "+";
        const key = layer + "_" + neuron + "_" + (sign === "-" ? "-" : "+");
        const data = __EXEMPLARS[key] || { isDead: true };
        return Promise.resolve(jsonResponse(JSON.stringify(data)));
      }

      // The Transluce util.js rewrites `./features/X` and (sometimes)
      // `./data/X` to Cloudfront. We never serve from the network in the
      // baked artifact, so synthesize a "no remote feature data" response.
      // For features/<scan>/<cantor>.json: try the exemplar map first
      // (decoded from cantor pair → (layer, neuron)); fall back to isDead.
      if (
        url.includes("d1fk9w8oratjix.cloudfront.net") ||
        url.includes("/features/") ||
        url.includes("/data/")
      ) {
        const m = url.match(/features\\/[^/]+\\/(\\d+)\\.json/);
        if (m) {
          // Cantor unpair: f = ((x+y)*(x+y+1))/2 + y, recover (x, y).
          const f = parseInt(m[1], 10);
          const w = Math.floor((Math.sqrt(8 * f + 1) - 1) / 2);
          const t = (w * w + w) / 2;
          const y = f - t;
          const x = w - y;
          for (const sign of ["+", "-"]) {
            const k = x + "_" + y + "_" + sign;
            if (__EXEMPLARS[k]) {
              return Promise.resolve(jsonResponse(JSON.stringify(__EXEMPLARS[k])));
            }
          }
        }
        return Promise.resolve(jsonResponse(JSON.stringify({ isDead: true })));
      }
    }
    return origFetch ? origFetch(input, opts) : Promise.reject(new Error("no fetch"));
  };
})();
</script>
"""


def bake_html(
    *,
    asset_root: Path,
    bundle_root: Path,
    out_html: Path,
    title: str | None = None,
    initial_slug: str | None = None,
) -> int:
    """Read the static frontend at `asset_root` (Transluce's
    circuits/frontend/assets/), pull in the data we wrote under
    `bundle_root/data/` and `bundle_root/graph_data/`, and emit a single
    self-contained HTML at `out_html`.

    Returns the byte size of the output file.
    """
    if not (asset_root / "index.html").exists():
        raise FileNotFoundError(f"missing {asset_root}/index.html")
    html = _read(asset_root / "index.html")

    # Collect all JSON data we need to embed.
    data_payload: dict[str, object] = {}
    meta_path = bundle_root / "data" / "graph-metadata.json"
    if meta_path.exists():
        data_payload["./data/graph-metadata.json"] = json.loads(_read(meta_path))
    for gp in sorted((bundle_root / "graph_data").glob("*.json")):
        slug = gp.stem
        data_payload[f"./graph_data/{slug}.json"] = json.loads(_read(gp))

    # Optional: neuron exemplars cache (slot key: "{layer}_{neuron}_{sign}").
    exemplars_payload: dict[str, object] = {}
    ex_path = bundle_root / "data" / "neuron_exemplars.json"
    if ex_path.exists():
        exemplars_payload = json.loads(_read(ex_path))

    bootstrap = (
        _BOOTSTRAP_TEMPLATE
        .replace("__DATA_PLACEHOLDER__", json.dumps(data_payload))
        .replace("__EXEMPLARS_PLACEHOLDER__", json.dumps(exemplars_payload))
    )

    html = _inline_styles_and_scripts(html, asset_root)

    # Inject the bootstrap so it runs BEFORE util.js's IIFE. Prefer right
    # after <head> if present; otherwise after the last <meta> in the
    # document head; otherwise before the first <link>/<script>.
    head_open = re.search(r"<head[^>]*>", html, flags=re.IGNORECASE)
    if head_open:
        idx = head_open.end()
        html = html[:idx] + "\n" + bootstrap + "\n" + html[idx:]
    else:
        # Find a good insertion point: after the last <meta> in the first
        # ~30 lines (i.e. inside the implicit head), but before any <link>
        # or <script>.
        # Take everything up to the first <link> or <script> as "head-ish",
        # then place after the last <meta> within that prefix.
        first_resource = re.search(r"<(link|script)\b", html, flags=re.IGNORECASE)
        prefix_end = first_resource.start() if first_resource else len(html)
        meta_iter = list(re.finditer(r"<meta[^>]*>", html[:prefix_end], flags=re.IGNORECASE))
        if meta_iter:
            idx = meta_iter[-1].end()
        else:
            doctype = re.search(r"<!DOCTYPE[^>]*>", html, flags=re.IGNORECASE)
            idx = doctype.end() if doctype else 0
        html = html[:idx] + "\n" + bootstrap + "\n" + html[idx:]

    if title:
        html = re.sub(
            r"<title>[^<]*</title>",
            f"<title>{title}</title>",
            html,
            count=1,
            flags=re.IGNORECASE,
        )

    if initial_slug:
        # The frontend reads `slug` from the URL query string. We can't
        # change the URL of a baked file, but we can pre-set
        # window.history.replaceState before util.js runs.
        slug_js = f"<script>history.replaceState(null, '', '?slug={initial_slug}');</script>\n"
        # Inject right after the bootstrap.
        html = html.replace(bootstrap, bootstrap + slug_js, 1)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html.stat().st_size
