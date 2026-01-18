from __future__ import annotations

from config.settings import Settings
from storage.cse_cache import CSECache
from clients.google_cse_client import GoogleCSEClient


def main() -> None:
    # 1) Load settings from .env
    settings = Settings.from_env(".env")
    cache = CSECache(settings.cache_dir)

    # 2) EXACT request (matches your cached JSON)
    request = {
        "key": settings.google_api_key,   # only used for cache key stripping; not stored
        "cx": "601e7542182d240af",
        "q": "تسليم مفتاح Riyadh",
        "start": 1,
        "num": 10,
        "hl": "en",
        "gl": "sa",
        "cr": "countrySA",
        "safe": "off",
    }

    # 3) Try cache
    cached = cache.get(request)
    if cached is not None:
        print("[CACHE HIT]")
        print("ok:", cached.ok)
        print("status_code:", cached.status_code)
        print("fetched_at:", cached.fetched_at)
        items = (cached.response or {}).get("items") or []
        print("items_count:", len(items))
        if items:
            # show first result briefly
            first = items[0]
            print("first_link:", first.get("link"))
            print("first_title:", first.get("title"))
        return

    # 4) Cache miss -> real API call (1 request)
    print("[CACHE MISS] -> calling Google API (1 request)")
    client = GoogleCSEClient(settings=settings, timeout=20.0, max_retries=2)
    res = client.search(
        q=request["q"],
        start=request["start"],
        num=request["num"],
        hl=request["hl"],
        gl=request["gl"],
        cr=request["cr"],
        safe=request["safe"],
    )

    print("api_ok:", res.ok)
    print("api_status_code:", res.status_code)
    if not res.ok:
        print("api_error:", res.error)
        return

    # Optional: show basic output
    data = res.data or {}
    items = data.get("items") or []
    print("api_items_count:", len(items))
    if items:
        print("api_first_link:", items[0].get("link"))
        print("api_first_title:", items[0].get("title"))


if __name__ == "__main__":
    main()
