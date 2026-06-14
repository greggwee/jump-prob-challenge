"""Phase B discovery: what WC 2026 match markets do Polymarket / Kalshi expose,
and in what shape? Public read-only endpoints, no auth."""

import json

import requests

HDR = {"User-Agent": "Mozilla/5.0 research"}


def probe_polymarket():
    print("=== POLYMARKET (gamma-api) ===")
    # events tagged for the World Cup; gamma supports text search via ?
    for url in [
        "https://gamma-api.polymarket.com/events?closed=false&limit=40&order=volume&ascending=false",
        "https://gamma-api.polymarket.com/markets?closed=false&limit=60&order=volume&ascending=false",
    ]:
        try:
            r = requests.get(url, headers=HDR, timeout=25)
            print(f"\nGET {url.split('?')[0]}  HTTP {r.status_code}")
            data = r.json()
            items = data if isinstance(data, list) else data.get("data", data)
            wc = [x for x in items if "world cup" in json.dumps(x).lower()
                  or "fifa" in json.dumps(x).lower()]
            print(f"  {len(items)} items, {len(wc)} mention world cup/fifa")
            sample = wc[:4] if wc else items[:2]
            for x in sample:
                q = x.get("title") or x.get("question") or x.get("slug")
                outs = x.get("outcomes") or x.get("markets")
                prices = x.get("outcomePrices")
                vol = x.get("volume") or x.get("volumeNum")
                print(f"    - {str(q)[:70]}")
                print(f"      keys={sorted(x.keys())[:12]}")
                if prices:
                    print(f"      outcomes={outs} prices={prices} vol={vol}")
        except Exception as e:
            print(f"  ERR: {str(e)[:150]}")


def probe_kalshi():
    print("\n=== KALSHI (trade-api v2) ===")
    for host in ["https://api.elections.kalshi.com", "https://trading-api.kalshi.com"]:
        try:
            r = requests.get(f"{host}/trade-api/v2/series?category=Sports",
                             headers=HDR, timeout=20)
            print(f"GET {host}/trade-api/v2/series?category=Sports  HTTP {r.status_code}")
            if r.status_code == 200:
                series = r.json().get("series", [])
                wc = [s for s in series if "world cup" in json.dumps(s).lower()
                      or "wcup" in json.dumps(s).lower() or "soccer" in json.dumps(s).lower()]
                print(f"  {len(series)} sports series; soccer/WC-ish: "
                      f"{[s.get('ticker') for s in wc][:10]}")
                break
        except Exception as e:
            print(f"  {host} ERR: {str(e)[:120]}")


probe_polymarket()
probe_kalshi()
