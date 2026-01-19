# Evaluation (Single Site Scoring)

This module turns a crawled website (or a single page) into a structured `SiteEvaluation` object with:
- **relevance** (is this a lead we want?)
- **score** (0–100)
- **confidence** (0.0–1.0)
- **lead_type + category** (e.g., `B2B`/`B2C`, `fit-out`, `showroom`, `architect`, `unknown`)
- **contacts** (emails/phones/WhatsApp)
- **signals + reasons** (debuggable explanation of why the score is what it is)

The evaluator is intentionally **heuristic + transparent**: every important decision should be visible through `signals` and `reasons`.

---

## What the evaluator predicts

For each input URL/site, we attempt to infer:

- **relevant** (`bool`): final classification
- **relevance_score** (`int`, 0–100): overall score
- **confidence** (`float`, 0–1): confidence in the classification
- **lead_type** (`str`): currently mostly `B2B` (extendable)
- **category** (`str`): e.g. `fit-out`, `showroom`, `architect`, `unknown`
- **company_name** (`str`): lightweight name inference (domain/title-like)
- **country / city** (`str`): geo inference (KSA-focused in current version)
- **contacts** (`dict`):
  - `emails: List[str]`
  - `phones: List[str]`
  - `whatsapp: List[str]`
  - `websites: List[str]`
- **source_type** (`str`): coarse page/site type used for penalties/guards:
  - `company_site`, `article`, `ecommerce`, `gov_edu`, ...
- **signals** (`dict[str, float]`): numeric debug features used by the scorer
- **reasons** (`List[str]`): human-readable explanation log

---

## Input format

### Primary API
The evaluator is used via `SiteEvaluator.evaluate(...)`.

Typical call pattern (as used in the pipeline):
1) Crawl up to N pages on the site (default N=5).
2) Combine content into one aggregated text blob.
3) Evaluate the site using URL + aggregated text.

Pipeline note (current code):
- `scripts/run_pipeline.py` calls `SiteCrawler.collect(...)` and passes `aggregated_text` into `evaluate(url, text=...)`.
- `html` is usually not provided in the pipeline path.

Conceptually:

- **url**: the canonical page URL being evaluated (usually the entrypoint)
- **text**: aggregated plain text from the crawler (recommended)
- **html**: optional raw HTML (if available); most scoring uses normalized text

Notes:
- The evaluator can work on a **single page**, but results are usually better with
  **aggregated multi-page text** from the crawler.
- If the extracted content is empty or near-empty, evaluation returns `relevant=False`
  with an explicit `empty_content` signal and reduced confidence.

---

## Output format (`SiteEvaluation`)

The evaluator returns a `SiteEvaluation` dataclass-like object.

Core fields:
- `url: str`
- `domain: str`
- `relevant: bool`
- `relevance_score: int`
- `confidence: float`
- `lead_type: str`
- `category: str`
- `contacts: Dict[str, List[str]]`
- `signals: Dict[str, float]`
- `reasons: List[str]`
- `source_type: str`
- `company_name: str`
- `country: str`
- `city: str`


---

## Evaluation logic (high level)

### 1) Normalize & extract features

* Normalize text (case-folding, whitespace cleanup, light tokenization).
* Count keyword hits in several buckets:

  * **high-precision (hp)**: strong service/industry phrases (best evidence)
  * **mid-precision (mp)**: related business/industry phrases
  * **business markers (biz)**: company-ish signals (services, contact, about, etc.)
  * **negatives (neg)**: hard/soft negatives (banking, travel, irrelevant verticals, etc.)
* Detect **geo** signals (KSA-focused): `.sa`, “Saudi/KSA”, Arabic tokens, `+966`, known cities.

### 2) Classify `source_type` (with guards)

We classify the page/site into a coarse type:

* `company_site`, `article`, `ecommerce`, `gov_edu`, ...
  This can affect scoring via penalties/guards.

**Article guard (homepage URLs):**
If the URL path looks like a homepage (`/`, `/en/`, `/ar/`, or <=1 path segment),
the evaluator blocks `article` classification even if blog-like tokens appear.

**Ecommerce override (anti-false-penalty):**
If the page looks “shop-like” but has strong B2B/portfolio evidence, we can override
`ecommerce → company_site` and remove the ecommerce penalty. This is tracked in:

* `signals["ecommerce_overridden"]`

B2B evidence used by the override includes any of:
- `portfolio_signal == 1`
- `fitout_signal == 1`
- `hp_count >= 6`
- explicit B2B terms like: `projects`, `case studies`, `clients`, `services`, `fit-out`, `turnkey`

### 3) Contact extraction

Extract and de-duplicate:

* emails (regex-based + normalization)
* phones (international formats; KSA detection uses `+966`)
* WhatsApp links/numbers

Contacts contribute to both:

* confidence (real businesses usually expose contacts)
* scoring (small positive bonus)

### 4) Apply gates (important)

Gates prevent false positives when text is weak/noisy.

* **KSA gate**: requires meaningful KSA evidence for KSA-focused mode.
* **Content gate**: requires enough strong/mid evidence to consider relevance.
* **Fallback content gate**: can pass if there is strong kitchen/fit-out evidence
  even when density is low.
  Current rule (all must be true):
  - `ksa_gate == 1`
  - contacts present (`email` or `phone`)
  - `mp_count >= 20`
  - AND one of:
    - `hp_count >= 2`, OR
    - `fitout_signal == 1`, OR
    - `kitchen_signal == 1` and `portfolio_signal == 1`
  - AND `neg_count < 40`

Empty/blocked content handling:

* If content is empty/near-empty (<= 30 words or < 200 chars), set `signals["empty_content"]=1`,
  lower confidence, force `content_gate=0`, and force non-relevant outcome.

### 5) Score + finalize

* Combine positives (hp/mp/biz, fit-out/interior/kitchen signals, contacts, geo)
  with negatives (hard negatives, heavy negative density).
* Cap negatives only when strong target evidence exists and no hard-negative hit
  (`signals["neg_capped"]`).
* Optionally apply targeted bonuses (e.g., showroom bonus when kitchen + contacts + low negatives).

Final:

* `relevance_score` in 0..100
* `relevant = (score >= threshold)` subject to gates/guards
* `confidence` based on signal strength and data quality

---

## Future Classification Development Note:

In future iterations, we can extend this evaluator into a **two-stage pipeline**. The current heuristic scorer should stay **fast and cheap**, primarily acting as a strong filter: it quickly accepts clearly relevant sites, rejects obvious non-target sites, and keeps a small “gray zone.” Only when the outcome is **ambiguous** (e.g., mid-range score and/or lower confidence, but with no hard-negative signals) we can escalate the site to an **LLM-based judge** with stricter instructions and richer context (aggregated text + extracted signals/contacts). This keeps costs under control while improving accuracy on borderline cases. We should **not** call an LLM for pages with **empty/blocked content** or clear hard-negative matches, since those cases are either undecidable from text or already confidently non-relevant.

