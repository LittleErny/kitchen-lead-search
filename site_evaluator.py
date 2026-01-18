# site_evaluator.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote
import httpx
import phonenumbers
import re
import html as html_lib

# -----------------------------
# Result model
# -----------------------------

@dataclass
class SiteEvaluation:
    url: str
    domain: str
    relevant: bool
    relevance_score: int
    confidence: float  # 0..1 (heuristic)
    lead_type: str = "unknown"   # "B2B" | "B2C" | "unknown"
    category: str = "unknown"    # showroom | designer | fit-out | developer | architect | individual | unknown
    contacts: Dict[str, List[str]] = field(default_factory=dict)
    signals: Dict[str, float] = field(default_factory=dict)  # debug-friendly features
    reasons: List[str] = field(default_factory=list)         # human-readable explanations


# -----------------------------
# Evaluator
# -----------------------------

class SiteEvaluator:
    """
    Rule-based site relevance evaluator.

    Input:
      - url (required)
      - html (optional) or text (optional). If both provided, html takes precedence.
      - You can extend keyword sets, weights, and thresholds via constructor.

    Output:
      - SiteEvaluation with score + reasons + tentative lead_type/category.
    """

    DEFAULT_THRESHOLDS = {
        "relevant": 70,     # score >= relevant => relevant
        "maybe_low": 45,    # between maybe_low..relevant => gray zone
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, int]] = None,
        keywords: Optional[Dict[str, List[str]]] = None,
        negative_keywords: Optional[List[str]] = None,
        cities_ksa: Optional[List[str]] = None,
    ):
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

        # weights: tune as you like
        self.weights = {
            "has_email": 8,
            "has_phone": 10,
            "has_whatsapp": 6,
            "has_address_hint": 6,
            "ksa_signal": 14,       # KSA = Kingdom of Saudi Arabia
            "kitchen_signal": 28,
            "interior_signal": 18,
            "fitout_signal": 22,
            "portfolio_signal": 10,
            "business_signal": 10,
            "negative_signal": -35,
            "non_ksa_penalty": -40,
        }
        if weights:
            self.weights.update(weights)

        # Keyword buckets
        self.keywords = self._default_keywords()
        if keywords:
            # merge/override
            for k, v in keywords.items():
                self.keywords[k] = v

        self.negative_keywords = negative_keywords or self._default_negative_keywords()

        # Minimal KSA cities list (can be expanded)
        self.cities_ksa = cities_ksa or [
            "riyadh", "jeddah", "dammam", "khobar", "al khobar", "makkah", "mecca",
            "medina", "madinah", "taif", "tabuk", "abha", "jizan", "najran", "jubail",
            "yanbu", "hail", "buraydah", "qassim", "al ahsa", "hofuf",
            # arabic variants (partial, best-effort)
            "الرياض", "جدة", "الدمام", "الخبر", "مكة", "المدينة", "الطائف",
        ]

    # ---------- Public API ----------

    def evaluate(self, url: str, html: Optional[str] = None, text: Optional[str] = None) -> SiteEvaluation:
        domain = self._domain(url)

        if html:
            extracted_text = self._html_to_text(html)
        else:
            extracted_text = text or ""

        # Normalize for matching
        norm_text = self._normalize(extracted_text)

        contacts = self._extract_contacts(extracted_text, url=url, html=html)


        signals: Dict[str, float] = {}
        reasons: List[str] = []

        # 1) Contact signals
        has_email = 1.0 if contacts.get("emails") else 0.0
        has_phone = 1.0 if contacts.get("phones") else 0.0
        has_whatsapp = 1.0 if contacts.get("whatsapp") else 0.0

        signals["has_email"] = has_email
        signals["has_phone"] = has_phone
        signals["has_whatsapp"] = has_whatsapp

        # 2) Geography (KSA)
        ksa_signal = self._ksa_signal(norm_text, contacts, domain)
        signals["ksa_signal"] = ksa_signal

        # If we see strong non-KSA signals, apply penalty
        non_ksa = self._non_ksa_signal(norm_text)
        signals["non_ksa"] = 1.0 if non_ksa else 0.0

        # 3) Industry signals
        kitchen_signal = self._bucket_signal(norm_text, self.keywords["kitchen"], min_hits=1)
        interior_signal = self._bucket_signal(norm_text, self.keywords["interior"], min_hits=1)

        # Require min 2 hits
        fitout_signal = self._bucket_signal(norm_text, self.keywords["fitout"], min_hits=2)
        portfolio_signal = self._bucket_signal(norm_text, self.keywords["portfolio"], min_hits=2)

        business_signal = self._bucket_signal(norm_text, self.keywords["business_markers"], min_hits=1)

        signals["kitchen_signal"] = kitchen_signal
        signals["interior_signal"] = interior_signal
        signals["fitout_signal"] = fitout_signal
        signals["portfolio_signal"] = portfolio_signal
        signals["business_signal"] = business_signal

        negative_signal = self._bucket_signal(norm_text, self.negative_keywords, min_hits=2)
        signals["negative_signal"] = negative_signal

        # 4) Compute score
        score = 0
        score += int(has_email * self.weights["has_email"])
        score += int(has_phone * self.weights["has_phone"])
        score += int(has_whatsapp * self.weights["has_whatsapp"])
        score += int(ksa_signal * self.weights["ksa_signal"])

        score += int(kitchen_signal * self.weights["kitchen_signal"])
        score += int(interior_signal * self.weights["interior_signal"])
        score += int(fitout_signal * self.weights["fitout_signal"])
        score += int(portfolio_signal * self.weights["portfolio_signal"])
        score += int(business_signal * self.weights["business_signal"])

        # Apply negative penalty only if the site does NOT look like our target industry.
        if negative_signal > 0 and (kitchen_signal == 0 and interior_signal == 0 and fitout_signal == 0):
            score += int(self.weights["negative_signal"])

        if non_ksa:
            score += int(self.weights["non_ksa_penalty"])

        # clamp 0..100 for readability
        score = max(0, min(100, score))

        # 5) Decide relevant + confidence
        relevant, confidence = self._decision(score)

        # 6) Tentative categorization (only if not a hard "no")
        category = "unknown"
        lead_type = "unknown"
        if score >= self.thresholds["maybe_low"]:
            category, lead_type = self._categorize(norm_text, contacts)

        # 7) Reasons (human readable)
        reasons.extend(self._build_reasons(signals, contacts, score, relevant, category, lead_type))

        return SiteEvaluation(
            url=url,
            domain=domain,
            relevant=relevant,
            relevance_score=score,
            confidence=confidence,
            lead_type=lead_type,
            category=category,
            contacts=contacts,
            signals=signals,
            reasons=reasons,
        )

    # ---------- Helpers: decision & reasons ----------

    def _decision(self, score: int) -> Tuple[bool, float]:
        if score >= self.thresholds["relevant"]:
            # high confidence grows with score
            return True, min(0.95, 0.70 + (score - self.thresholds["relevant"]) / 100.0)
        if score < self.thresholds["maybe_low"]:
            # low confidence "no" grows as score approaches 0
            return False, min(0.90, 0.60 + (self.thresholds["maybe_low"] - score) / 100.0)
        # gray zone
        return False, 0.55

    def _build_reasons(
        self,
        signals: Dict[str, float],
        contacts: Dict[str, List[str]],
        score: int,
        relevant: bool,
        category: str,
        lead_type: str,
    ) -> List[str]:
        r: List[str] = []
        r.append(f"Score={score} => {'RELEVANT' if relevant else 'NOT SURE/NOT RELEVANT'}; category={category}; lead_type={lead_type}")

        if contacts.get("emails"):
            r.append(f"Found emails: {contacts['emails'][:3]}")
        if contacts.get("phones"):
            r.append(f"Found phones: {contacts['phones'][:3]}")
        if contacts.get("whatsapp"):
            r.append("Found WhatsApp link/number")

        if signals.get("ksa_signal", 0) > 0:
            r.append("KSA signal detected (city/ksa/saudi/+966)")
        if signals.get("non_ksa", 0) > 0:
            r.append("Strong non-KSA signal detected (possible outside target region)")

        if signals.get("kitchen_signal", 0) > 0:
            r.append("Kitchen/cabinetry keywords detected")
        if signals.get("fitout_signal", 0) > 0:
            r.append("Fit-out/finishing keywords detected")
        if signals.get("interior_signal", 0) > 0:
            r.append("Interior design keywords detected")
        if signals.get("portfolio_signal", 0) > 0:
            r.append("Portfolio/projects keywords detected")
        if signals.get("negative_signal", 0) > 0:
            r.append("Negative/irrelevant keywords detected")

        return r

    # ---------- Helpers: categorization ----------

    def _categorize(self, norm_text: str, contacts: Dict[str, List[str]]) -> Tuple[str, str]:
        """
        Lightweight categorization + lead type inference.

        Key rules:
          - 'individual' category is only allowed for true B2C-like pages
            (individual markers strong AND no business markers AND no corporate contacts).
          - Prefer architect over fit-out on ties (common for engineering offices).
          - Keep category and lead_type consistent (no 'individual' with B2B).
        """

        # --- Helper: call bucket signal with optional min_hits (works with both old/new implementations)
        def sig(bucket_name: str, min_hits: int = 2) -> float:
            try:
                return self._bucket_signal(norm_text, self.keywords[bucket_name], min_hits=min_hits)  # type: ignore
            except TypeError:
                # Fallback if your _bucket_signal doesn't accept min_hits yet
                return self._bucket_signal(norm_text, self.keywords[bucket_name])  # type: ignore

        def sig_terms(terms: List[str], min_hits: int = 2) -> float:
            try:
                return self._bucket_signal(norm_text, terms, min_hits=min_hits)  # type: ignore
            except TypeError:
                return self._bucket_signal(norm_text, terms)  # type: ignore

        # --- Base signals for lead type decision
        business_val = sig_terms(self.keywords["business_markers"], min_hits=1)

        # Make "individual" harder to trigger: require 2+ hits (or strict bucket)
        individual_val = sig_terms(self.keywords["individual_markers"], min_hits=2)

        # Corporate contact hint: email/phone present usually means company-like (not always, but good heuristic)
        has_contacts = bool(contacts.get("emails") or contacts.get("phones"))

        # Decide lead_type first (helps to gate 'individual')
        # B2C only if individual markers are strong AND there is no company signal
        if individual_val > 0 and business_val == 0 and not has_contacts:
            lead_type = "B2C"
        else:
            lead_type = "B2B"

        # --- Category scores (use stricter thresholds for noisy buckets)
        scores = {
            "showroom": sig("showroom", min_hits=1),
            "fit-out": sig("fitout", min_hits=2),  # require more evidence
            "designer": sig("designer", min_hits=1),
            "architect": sig("architect", min_hits=1),
            "developer": sig("developer", min_hits=1),
            # only allow individual if lead_type is B2C (gate)
            "individual": individual_val if lead_type == "B2C" else 0.0,
        }

        top_val = max(scores.values()) if scores else 0.0
        if top_val <= 0:
            return ("unknown", lead_type)

        # Tie-break priority:
        # - prefer architect over fit-out when equal (engineering offices often mention finishing/turnkey)
        # - showroom next, then designer, developer, individual
        priority = ["architect", "fit-out", "showroom", "designer", "developer", "individual"]
        category = next(cat for cat in priority if scores.get(cat, 0.0) == top_val)

        # Consistency rule: if somehow individual selected but lead_type is B2B, fix it
        if category == "individual" and lead_type != "B2C":
            category = "unknown"

        return category, lead_type

    # ---------- Helpers: signals ----------

    def _bucket_signal(self, norm_text: str, terms: List[str], min_hits: int = 2) -> float:
        hits = 0
        for t in terms:
            if t in norm_text:
                hits += 1
                if hits >= min_hits:
                    return 1.0
        return 0.0

    def _ksa_signal(self, norm_text: str, contacts: Dict[str, List[str]], domain: str) -> float:
        # phone prefix +966 is a strong hint
        if any(p.startswith("+966") for p in contacts.get("phones", [])):
            return 1.0

        # country keywords
        if "saudi" in norm_text or "ksa" in norm_text or "السعود" in norm_text:
            return 1.0

        # city mentions
        for c in self.cities_ksa:
            if self._normalize(c) in norm_text:
                return 0.8

        # domain hint (.sa)
        if domain.endswith(".sa"):
            return 0.7

        return 0.0

    def _non_ksa_signal(self, norm_text: str) -> bool:
        # Very rough: penalize if explicit other countries appear strongly
        non_ksa_terms = [
            "uae", "dubai", "abu dhabi", "qatar", "doha", "kuwait",
            "oman", "bahrain", "egypt", "cairo", "turkey", "istanbul",
            "uk", "london", "usa", "new york", "germany", "berlin",
        ]
        # If both KSA and non-KSA present, we won't penalize
        if ("saudi" in norm_text) or ("ksa" in norm_text) or ("السعود" in norm_text):
            return False
        for t in non_ksa_terms:
            if t in norm_text:
                return True
        return False

    # ---------- Helpers: contacts & text ----------

    def _extract_contacts(self, text: str, url: str, html: Optional[str] = None) -> Dict[str, List[str]]:
        emails = set(self._find_emails(text))
        phones = set(self._find_phones(text))
        whatsapp = set(self._find_whatsapp(text))

        # NEW: also scan HTML attributes (mailto:, tel:, whatsapp links, visible obfuscations)
        if html:
            emails |= set(self._find_emails_in_html_attrs(html))
            phones |= set(self._find_phones_in_html_attrs(html))
            whatsapp |= set(self._find_whatsapp_in_html_attrs(html))

        websites = []
        try:
            websites = [self._domain(url)]
        except Exception:
            pass

        return {
            "emails": sorted(emails),
            "phones": sorted(phones),
            "whatsapp": sorted(whatsapp),
            "websites": websites,
        }

    def _find_emails(self, text: str) -> List[str]:
        # handles normal emails + "name (at) domain.com"
        cleaned = text.replace("(at)", "@").replace("[at]", "@").replace(" at ", "@")
        pattern = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
        return [m.group(0).lower() for m in pattern.finditer(cleaned)]

    def _find_emails_in_html_attrs(self, html: str) -> List[str]:
        # decode entities and percent-encoding so mailto:info%40a.com works
        raw = html_lib.unescape(html)
        raw = unquote(raw)

        emails: set[str] = set()

        # mailto:info@domain.com?subject=...
        for m in re.finditer(r'(?i)\bmailto:\s*([^"\'\s>]+)', raw):
            chunk = m.group(1)
            chunk = chunk.split("?")[0]  # cut params
            # may contain multiple separated by comma/semicolon
            for part in re.split(r"[,\s;]+", chunk):
                part = part.strip()
                if not part:
                    continue
                # validate via existing regex
                emails.update(self._find_emails(part))

        # some sites store emails in attributes without mailto
        # e.g. data-email="info@x.com"
        # We'll just search emails in all attribute blobs:
        # (cheap and effective; your email regex will filter)
        attr_vals = re.findall(r'(?is)\b(?:href|data-email|data-mail|data-contact)\s*=\s*["\']([^"\']+)["\']', raw)
        for v in attr_vals:
            emails.update(self._find_emails(v))

        return sorted(emails)

    def _find_phones(self, text: str) -> List[str]:
        # Candidate chunks that look like phone-like sequences
        candidates = re.findall(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
        out = set()

        for raw in candidates:
            raw = raw.strip()
            try:
                # If no +, assume Saudi region (SA). If +, region None.
                num = phonenumbers.parse(raw, None if raw.startswith("+") else "SA")
                if not phonenumbers.is_possible_number(num):
                    continue
                if not phonenumbers.is_valid_number(num):
                    continue

                e164 = phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
                digits = re.sub(r"\D", "", e164)
                if 10 <= len(digits) <= 15:
                    out.add(e164)
            except phonenumbers.NumberParseException:
                continue

        return sorted(out)

    def _find_phones_in_html_attrs(self, html: str) -> List[str]:
        raw = html_lib.unescape(html)
        raw = unquote(raw)

        phones: set[str] = set()

        # tel:+966...
        for m in re.finditer(r'(?i)\btel:\s*([^"\'\s>]+)', raw):
            chunk = m.group(1).split("?")[0]
            phones.update(self._find_phones(chunk))

        # also scan typical attrs that contain phones
        attr_vals = re.findall(r'(?is)\b(?:href|data-phone|data-tel|data-contact)\s*=\s*["\']([^"\']+)["\']', raw)
        for v in attr_vals:
            phones.update(self._find_phones(v))

        return sorted(phones)

    def _find_whatsapp(self, text: str) -> List[str]:
        # WhatsApp links or "wa.me/..."
        pattern = re.compile(r"(https?://(?:wa\.me|api\.whatsapp\.com)/[^\s\"'<]+)", re.IGNORECASE)
        return [m.group(1) for m in pattern.finditer(text)]

    def _find_whatsapp_in_html_attrs(self, html: str) -> List[str]:
        raw = html_lib.unescape(html)
        raw = unquote(raw)

        wa: set[str] = set()

        # reuse your existing extractor on the raw html
        wa.update(self._find_whatsapp(raw))

        # also try to catch common formats in href:
        hrefs = re.findall(r'(?is)\bhref\s*=\s*["\']([^"\']+)["\']', raw)
        for h in hrefs:
            wa.update(self._find_whatsapp(h))

        return sorted(wa)

    def _html_to_text(self, html: str) -> str:
        # Remove scripts/styles
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        # Remove tags
        html = re.sub(r"(?s)<[^>]+>", " ", html)
        # Unescape basic entities (minimal)
        html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", "\"").replace("&lt;", "<").replace("&gt;", ">")
        # Collapse whitespace
        return re.sub(r"\s+", " ", html).strip()

    def _normalize(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def _domain(self, url: str) -> str:
        p = urlparse(url)
        host = p.netloc or ""
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    # ---------- Keyword defaults ----------

    def _default_keywords(self) -> Dict[str, List[str]]:
        return {
            # Core relevance buckets
            "kitchen": [
                # EN
                "kitchen", "kitchens", "kitchen design", "kitchen installation",
                "kitchen supplier", "kitchen company", "kitchen showroom",
                "cabinet", "cabinets", "cabinetry", "kitchen cabinets",
                "wardrobe", "wardrobes", "closet", "closets",
                "fitted wardrobe", "built-in wardrobe", "walk-in closet",
                "joinery", "woodwork", "carpentry", "millwork",
                "modular kitchen", "custom kitchen", "custom cabinetry",
                "countertop", "worktop", "granite", "quartz", "marble",
                "vanity", "vanities", "tv unit", "wall unit",

                # AR
                "مطابخ", "مطبخ", "تصميم مطابخ", "تركيب مطابخ", "تنفيذ مطابخ",
                "شركة مطابخ", "معرض مطابخ",
                "خزائن", "خزانة", "خزائن مطبخ", "دواليب", "دولاب",
                "غرف ملابس", "غرفة ملابس", "كلوزيت", "ووك إن كلوزيت",
                "نجارة", "أعمال نجارة", "أعمال خشبية", "تفصيل",
                "مطابخ مودرن", "مطابخ تفصيل", "تفصيل مطابخ",
                "رخام", "جرانيت", "كوارتز", "سطح", "كونترتوب",
                "مغاسل", "وحدات حمام", "وحدة تلفزيون", "وحدة حائط",
            ],

            "interior": [
                # EN
                "interior", "interiors", "interior design", "interior designer",
                "design studio", "design consultancy", "space planning",
                "decoration", "decor", "home decor", "styling",
                "furniture design", "custom furniture", "bespoke furniture",
                "residential interior", "commercial interior",

                # AR
                "تصميم داخلي", "مصمم داخلي", "استشارات تصميم", "استشارات ديكور",
                "استديو تصميم", "تخطيط المساحات", "توزيع المساحات",
                "ديكور", "ديكورات", "ديكور منزلي", "تنسيق",
                "تصميم أثاث", "أثاث تفصيل", "تفصيل أثاث", "أثاث حسب الطلب",
                "تصميم سكني", "تصميم تجاري",
            ],

            "fitout": [
                # EN
                "fit-out", "fit out", "fitout", "interior fit-out",
                "turnkey", "turn key", "turn-key", "turnkey solutions",
                "interior finishing", "construction finishing",
                "villa finishing", "office fit-out", "commercial fit-out",
                "renovation", "remodel", "refurbishment",
                "contracting", "contractor", "general contractor",
                "civil works", "gypsum", "false ceiling", "partition",
                "mep", "electrical", "plumbing", "hvac",

                # AR
                "فيت اوت", "فت اوت", "تشطيب", "تشطيبات", "تشطيب داخلي",
                "تسليم مفتاح", "حلول متكاملة", "تنفيذ متكامل",
                "ترميم", "تجديد", "إعادة تأهيل", "تعديل",
                "مقاولات", "مقاول", "مقاول عام", "شركة مقاولات",
                "أعمال مدنية", "جبس", "جبس بورد", "أسقف مستعارة",
                "قواطع", "مقسمات",
                "ميكانيكا", "كهرباء", "سباكة", "تكييف", "تمديدات",
                "إشراف", "إدارة مشروع",
            ],

            "portfolio": [
                # EN
                "portfolio", "projects", "case study", "case studies",
                "gallery", "project gallery", "before and after",
                "completed projects", "recent projects",

                # AR
                "مشاريع", "مشروع", "اعمالنا", "أعمالنا", "نماذج أعمال",
                "معرض اعمال", "معرض أعمال", "ألبوم", "قبل وبعد",
                "مشاريع منفذة", "أحدث المشاريع",
            ],

            # Categorization buckets
            "showroom": [
                # EN
                "showroom", "kitchen showroom", "display", "exhibition",
                "experience center", "store", "shop",

                # AR
                "معرض", "صالة عرض", "مركز تجربة", "متجر", "محل",
                "عرض", "معرض مطابخ",
            ],

            "designer": [
                # EN
                "interior designer", "interior design", "design consultancy",
                "space planning", "concept design", "moodboard",

                # AR
                "مصمم داخلي", "تصميم داخلي", "استشارات تصميم",
                "تخطيط المساحات", "تصميم مفاهيمي", "لوحة إلهام",
            ],

            "architect": [
                # EN
                "architect", "architecture", "architectural", "architects",
                "engineering office", "engineering consultancy",
                "architectural design", "structural design",
                "site supervision", "permit", "building permit",

                # AR
                "معماري", "هندسة معمارية", "تصميم معماري", "مكتب هندسي",
                "استشارات هندسية", "تصميم إنشائي", "هندسة إنشائية",
                "إشراف", "إشراف هندسي", "رخصة بناء", "تصريح بناء",
                "مخططات", "مخططات معمارية", "اعتماد مخططات",
            ],

            "developer": [
                # EN
                "real estate developer", "property developer", "development company",
                "real estate development", "projects development",

                # AR
                "مطوّر عقاري", "مطور عقاري", "تطوير عقاري", "شركة تطوير",
                "تطوير مشاريع", "تطوير عقاري سكني", "تطوير عقاري تجاري",
            ],

            "individual_markers": [
                # EN
                "i need", "looking for", "need a", "want to", "can someone",
                "my villa", "my apartment", "my house", "my kitchen",
                "budget", "price", "how much", "cost",
                "whatsapp me", "call me", "dm me", "contact me",

                # AR
                "احتاج", "أحتاج", "ابحث عن", "أبحث عن", "محتاج", "اريد", "أريد",
                "بيتي", "شقتي", "فيلا", "مطبخي", "مطبخ",
                "ميزانية", "سعر", "اسعار", "تكلفة", "كم السعر", "كم يكلف",
                "واتساب", "راسلني", "اتصل علي", "كلموني", "رسالة خاصة",
            ],

            "business_markers": [
                # EN
                "llc", "ltd", "company", "co.", "corp", "inc",
                "contracting", "contractor", "trading", "factory", "manufacturer",
                "since", "est.", "established", "branches", "head office",
                "commercial", "b2b", "profile", "company profile",
                "iso", "vat", "cr number", "registration",

                # AR
                "شركة", "مؤسسة", "مجموعة", "مصنع", "شركة تصنيع",
                "مقاول", "مقاولات", "شركة مقاولات", "مقاول عام",
                "تجارة", "تجارية", "توريد",
                "تأسست", "تأسيس", "منذ", "فرع", "فروع", "المقر الرئيسي",
                "سجل تجاري", "رقم السجل", "رقم ضريبي", "الرقم الضريبي",
                "بروفايل", "ملف الشركة", "شهادة", "ايزو",
                "مكتب هندسي", "استشارات هندسية",
            ],
        }

    def _default_negative_keywords(self) -> List[str]:
        return [
            # EN (irrelevant areas)
            "car repair", "automotive", "garage", "tyres", "tire",
            "restaurant", "cafe", "bakery", "catering",
            "hospital", "clinic", "dentist", "pharmacy",
            "hotel", "booking", "travel agency", "tourism",
            "crypto", "forex", "bitcoin", "trading platform",
            "used cars", "car dealer",
            "hair salon", "barber", "spa",

            # AR (irrelevant areas)
            "ورشة سيارات", "صيانة سيارات", "ميكانيكا", "كهرباء سيارات", "إطارات",
            "مطعم", "كافيه", "مقهى", "مخبز", "تموين",
            "مستشفى", "عيادة", "طبيب أسنان", "صيدلية",
            "فندق", "حجز", "سفر", "وكالة سفر", "سياحة",
            "عملات رقمية", "تداول", "فوركس", "بيتكوين", "منصة تداول",
            "سيارات مستعملة", "معرض سيارات",
            "صالون", "حلاق", "سبا",
        ]

    def evaluate_url(self, url: str, timeout: float = 15.0) -> SiteEvaluation:
        html = self._fetch_html(url, timeout=timeout)
        return self.evaluate(url, html=html)

    def _fetch_html(self, url: str, timeout: float = 15.0) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        }
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            r = client.get(url, headers=headers)
            r.raise_for_status()
            return r.text


# -----------------------------
# Example usage (manual testing)
# -----------------------------
if __name__ == "__main__":
    # demo_text = """
    # Al Noor Interiors - Riyadh, Saudi Arabia.
    # We provide kitchen cabinets, custom wardrobes, and full fit-out solutions for villas and offices.
    # Contact: info@alnoor.sa | +966 55 123 4567
    # Portfolio: Our Projects
    # """
    url = "https://fayunited.com.sa"

    ev = SiteEvaluator().evaluate_url(url=url)
    print(ev)
    for reason in ev.reasons:
        print("-", reason)
