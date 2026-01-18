# site_crawler.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

from crawling.fetcher import CachedFetcher, FetchResult


@dataclass
class AggregatedContent:
    base_url: str
    visited_urls: List[str] = field(default_factory=list)
    failed_urls: List[Tuple[str, str]] = field(default_factory=list)  # (url, error)
    aggregated_text: str = ""


class _LinkExtractor(HTMLParser):
    """
    Very small HTML link extractor using stdlib HTMLParser.
    Captures href + anchor text.
    """
    def __init__(self):
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._current_href: Optional[str] = None
        self._current_text_chunks: List[str] = []
        self._in_a = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = None
            for k, v in attrs:
                if k.lower() == "href":
                    href = v
                    break
            if href:
                self._in_a = True
                self._current_href = href
                self._current_text_chunks = []

    def handle_data(self, data):
        if self._in_a and data:
            self._current_text_chunks.append(data.strip())

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_a:
            text = " ".join([t for t in self._current_text_chunks if t])
            self.links.append((self._current_href or "", text))
            self._in_a = False
            self._current_href = None
            self._current_text_chunks = []


class SiteCrawler:
    """
    Depth-1 "smart collector":
      - fetch home
      - extract internal links
      - pick up to one best page from each bucket: contact/about/services/projects
      - fetch those pages (max_pages)
      - return aggregated text

    Keeps logic simple and predictable, to avoid CAPTCHAs and runaway crawling.
    """

    def __init__(
        self,
        fetcher: CachedFetcher,
        max_pages: int = 5,
        max_links_scanned: int = 250,
        include_home: bool = True,
    ):
        self.fetcher = fetcher
        self.max_pages = max_pages
        self.max_links_scanned = max_links_scanned
        self.include_home = include_home

        # Patterns in both EN + AR
        self.bucket_patterns: Dict[str, List[str]] = {
            "contact": [
                "contact", "contacts", "get-in-touch", "call-us",
                "اتصل", "إتصل", "اتصل بنا", "تواصل", "تواصل معنا", "للتواصل",
            ],
            "about": [
                "about", "about-us", "who-we-are", "company", "profile", "our-story",
                "من نحن", "نبذة", "نبذة عنا", "عن", "تعريف", "ملف الشركة",
            ],
            "services": [
                "services", "service", "solutions", "what-we-do",
                "خدمات", "خدماتنا", "حلول", "مجالات",
            ],
            "projects": [
                "projects", "project", "portfolio", "gallery", "our-work", "work",
                "مشاريع", "مشروع", "اعمالنا", "أعمالنا", "معرض", "صور", "ألبوم",
            ],
        }

    # ---------- Public API ----------

    def collect(self, base_url: str) -> AggregatedContent:
        base_url = base_url.strip()
        out = AggregatedContent(base_url=base_url)

        home_res = self.fetcher.fetch_text(base_url)
        if not home_res.ok:
            out.failed_urls.append((base_url, home_res.error or f"HTTP {home_res.status_code}"))
            out.aggregated_text = ""
            return out

        # Decide which pages to fetch (home + buckets)
        candidates = self._extract_internal_links(home_res.final_url, home_res.text)
        picked = self._pick_priority_pages(home_res.final_url, candidates)

        urls_to_fetch: List[str] = []
        if self.include_home:
            urls_to_fetch.append(home_res.final_url)

        # Add picked bucket pages
        for u in picked:
            if u not in urls_to_fetch:
                urls_to_fetch.append(u)

        # Clamp to max_pages
        urls_to_fetch = urls_to_fetch[: self.max_pages]

        # Fetch and aggregate
        texts: List[str] = []
        for u in urls_to_fetch:
            res = home_res if u == home_res.final_url else self.fetcher.fetch_text(u)
            if not res.ok:
                out.failed_urls.append((u, res.error or f"HTTP {res.status_code}"))
                continue

            # Keep only HTML-ish content
            ctype = (res.content_type or "").lower()
            if ctype and ("text/html" not in ctype and "application/xhtml" not in ctype and "text/" not in ctype):
                # Some servers mislabel; we still try, but skip obvious binaries
                if "pdf" in ctype or "image" in ctype or "octet-stream" in ctype:
                    continue

            page_text = self._html_to_text(res.text)
            if page_text:
                texts.append(page_text)
                out.visited_urls.append(res.final_url)

        out.aggregated_text = "\n".join(texts).strip()
        return out

    # ---------- Link selection ----------

    def _extract_internal_links(self, base_url: str, html: str) -> List[Tuple[str, str]]:
        # Parse links
        parser = _LinkExtractor()
        try:
            parser.feed(html)
        except Exception:
            return []

        links = parser.links[: self.max_links_scanned]
        base_domain = self._domain(base_url)

        internal: List[Tuple[str, str]] = []
        for href, anchor in links:
            if not href:
                continue
            abs_url = urljoin(base_url, href)
            abs_url = self._normalize_candidate_url(abs_url)

            if not abs_url:
                continue
            if self._domain(abs_url) != base_domain:
                continue

            internal.append((abs_url, anchor or ""))

        # de-dup while preserving order
        seen = set()
        deduped: List[Tuple[str, str]] = []
        for u, a in internal:
            if u in seen:
                continue
            seen.add(u)
            deduped.append((u, a))
        return deduped

    def _pick_priority_pages(self, base_url: str, candidates: List[Tuple[str, str]]) -> List[str]:
        """
        Pick best page per bucket by scoring URL + anchor.
        """
        # Score each candidate per bucket
        best: Dict[str, Tuple[int, str]] = {}  # bucket -> (score, url)

        for url, anchor in candidates:
            u_low = (url or "").lower()
            a_low = (anchor or "").lower()

            for bucket, patterns in self.bucket_patterns.items():
                score = 0
                for p in patterns:
                    p_low = p.lower()
                    if p_low in u_low:
                        score += 100
                    if p_low in a_low:
                        score += 80

                # Extra minor heuristics
                if bucket == "contact" and ("whatsapp" in u_low or "wa.me" in u_low):
                    score += 20
                if bucket == "projects" and ("gallery" in u_low or "portfolio" in u_low):
                    score += 10

                if score <= 0:
                    continue

                cur = best.get(bucket)
                if cur is None or score > cur[0]:
                    best[bucket] = (score, url)

        # Return in a sensible order
        ordered_buckets = ["contact", "about", "services", "projects"]
        picked = [best[b][1] for b in ordered_buckets if b in best]

        # Fallback: only when we found no candidates at all
        if not picked and not candidates:
            fallbacks = [
                urljoin(base_url, "contact"),
                urljoin(base_url, "contact-us"),
                urljoin(base_url, "about"),
                urljoin(base_url, "services"),
                urljoin(base_url, "projects"),
            ]
            picked = [self._normalize_candidate_url(u) for u in fallbacks if self._normalize_candidate_url(u)]

        # De-dup
        uniq = []
        seen = set()
        for u in picked:
            if u and u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq

    # ---------- HTML to text ----------

    def _html_to_text(self, html: str) -> str:
        if not html:
            return ""

        # remove script/style blocks
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        html = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", html)

        # remove comments
        html = re.sub(r"(?is)<!--.*?-->", " ", html)

        # strip tags
        html = re.sub(r"(?s)<[^>]+>", " ", html)

        # basic entities
        html = (
            html.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&lt;", "<")
            .replace("&gt;", ">")
        )

        # collapse whitespace
        html = re.sub(r"\s+", " ", html).strip()

        # keep it bounded (avoid megabytes of text)
        if len(html) > 200_000:
            html = html[:200_000]
        return html

    # ---------- URL helpers ----------

    def _normalize_candidate_url(self, url: str) -> str:
        if not url:
            return ""
        url = url.strip()

        # drop fragments
        url, _frag = urldefrag(url)

        # ignore obvious non-pages
        lower = url.lower()
        if lower.startswith("mailto:") or lower.startswith("tel:") or lower.startswith("javascript:"):
            return ""
        if any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip", ".rar", ".7z", ".pdf"]):
            return ""
        return url

    def _domain(self, url: str) -> str:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
