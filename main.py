from fetcher import CachedFetcher
from site_crawler import SiteCrawler
from site_evaluator import SiteEvaluator

fetcher = CachedFetcher(cache_dir=".cache/http", min_delay=0.5, max_delay=1.3, max_retries=3)
crawler = SiteCrawler(fetcher, max_pages=5)
evaluator = SiteEvaluator()

url = "https://www.point5kitchens.co.uk/"

agg = crawler.collect(url)
ev = evaluator.evaluate(url, text=agg.aggregated_text)

print("Visited:", "\n".join(agg.visited_urls))
print("Failed:", agg.failed_urls)
print(ev.relevance_score, ev.category, ev.lead_type)
for r in ev.reasons:
    print("-", r)

fetcher.close()
