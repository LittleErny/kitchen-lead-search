# evaluator.py
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import html as html_lib
import httpx
import phonenumbers

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
    decision: str = "reject"  # accept | review | reject

    # Existing fields (keep)
    lead_type: str = "unknown"   # "B2B" | "B2C" | "unknown"
    category: str = "unknown"    # showroom | designer | fit-out | developer | architect | individual | unknown
    contacts: Dict[str, List[str]] = field(default_factory=dict)
    signals: Dict[str, float] = field(default_factory=dict)  # debug-friendly features
    reasons: List[str] = field(default_factory=list)         # human-readable explanations

    source_type: str = "unknown"     # company_site | marketplace | article | ecommerce | retail_catalog | gov_edu | unknown
    company_name: str = "unknown"
    country: str = "unknown"
    city: str = "unknown"
    empty_content: bool = False


# -----------------------------
# Evaluator
# -----------------------------


class SiteEvaluator:
    """
    Rule-based site relevance evaluator.

    Interface compatibility:
      - evaluate(url, html=None, text=None) -> SiteEvaluation
      - evaluate_url(url, timeout=15.0) -> SiteEvaluation
      - keeps keyword customization & thresholds/weights constructor params
    """

    DEFAULT_THRESHOLDS = {
        "relevant": 70,     # (score >= relevant) means relevant
        "maybe_low": 45,    # between maybe_low and relevant means gray zone
    }

    # -----------------------------
    # Init
    # -----------------------------

    def __init__(
        self,
        thresholds: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, int]] = None,
        keywords: Optional[Dict[str, List[str]]] = None,
        negative_keywords: Optional[List[str]] = None,
        cities_ksa: Optional[List[str]] = None,
        mode: str = "precision",
    ):
        mode = (mode or "precision").strip().lower()
        if mode not in ("precision", "recall"):
            raise ValueError(f"Invalid evaluator mode: {mode}")
        self.mode = mode

        # Thresholds
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

        # Weights (kept for backward-compat; not all are used directly anymore)
        self.weights = {
            # contact confirmation (small bonuses)
            "has_email": 3,
            "has_phone": 4,
            "has_whatsapp": 2,

            # geography
            "ksa_bonus": 12,
            "non_ksa_penalty": -35,

            # core scoring blocks
            "hp_max": 35,          # high precision max contribution
            "mp_max": 18,          # mid precision max contribution
            "biz_max": 15,         # business-intent max contribution
            "density_max": 20,     # density max contribution
            "negative_max": -35,   # negative max penalty by keyword-spotting (bigger penalty is possible by page-types)

            # page-type penalties
            "penalty_gov_edu": -80,
            "penalty_article": -35,
            "penalty_marketplace": -35,
            "penalty_ecommerce": -12,  # mild (some ecommerce is relevant)
            "penalty_retail_catalog": -25,
        }
        if weights:
            self.weights.update(weights)

        # Keywords: default buckets + allow override/merge
        self.keywords = self._default_keywords()
        if keywords:
            for k, v in keywords.items():
                self.keywords[k] = v

        self.negative_keywords = negative_keywords or self._default_negative_keywords()
        self.hard_negative_terms = [
            "bank", "banking", "loan", "credit card", "insurance",
            "pest control", "exterminator", "termite",
            "water leaks", "leak detection", "leak repair", "plumbing leak",
            "cleaning services", "house cleaning", "general cleaning",
            "car", "cars", "auto", "automotive", "dealership", "car dealer",
            "travel", "hotels", "flight booking", "ticketing",
            "food delivery", "restaurant delivery",
            "real estate listings", "property listings", "classifieds",
            "riyadbank", "riyadh bank",
            "ورشة سيارات", "صيانة سيارات", "مستشفى", "عيادة",
            "بنك", "قرض", "بطاقة", "تأمين",
            "مكافحة حشرات", "نمل", "صراصير",
            "كشف تسرب", "تسربات المياه", "تسربات", "كشف تسريب",
            "تنظيف", "شركة تنظيف", "مغاسل", "غسيل",
            "سيارات", "معرض سيارات", "بيع سيارات",
            "سفر", "حجوزات", "طيران", "تذاكر",
            "توصيل الطعام", "طلبات الطعام",
            "عقار", "عقارات", "إعلانات عقارية",
        ]

        # Cities list (KSA)
        self.cities_ksa = cities_ksa or [
            "riyadh", "jeddah", "dammam", "khobar", "al khobar", "makkah", "mecca",
            "medina", "madinah", "taif", "tabuk", "abha", "jizan", "najran", "jubail",
            "yanbu", "hail", "buraydah", "qassim", "al ahsa", "hofuf",
            "الرياض", "جدة", "الدمام", "الخبر", "مكة", "مكة المكرمة",
            "المدينة", "المدينة المنورة", "الطائف", "تبوك", "أبها", "جيزان",
            "نجران", "الجبيل", "ينبع", "حائل", "بريدة", "القصيم",
            "الأحساء", "الاحساء", "الهفوف",
        ]

        # Precompile regex patterns for performance & correctness
        self._compiled_terms_cache: Dict[str, List[re.Pattern]] = {}

    # -----------------------------
    # Public API
    # -----------------------------

    def evaluate(self, url: str, html: Optional[str] = None, text: Optional[str] = None) -> SiteEvaluation:
        domain = self._domain(url)

        # Extract text
        if html:
            body_text = text or self._html_to_text(html)
            sections = self._extract_sections(html)
            jsonld = self._extract_jsonld(html)
        else:
            body_text = text or ""
            sections = {"title": "", "meta": "", "h1": "", "h2": "", "footer": "", "contact": ""}
            jsonld = {}

        # Contacts
        contacts = self._extract_contacts(body_text, url=url, html=html)

        # Normalize
        norm_body = self._normalize(body_text)
        norm_sections = {k: self._normalize(v) for k, v in sections.items()}

        # Source type (page type)
        source_type, article_guard_blocked = self._classify_source_type(url, domain, norm_body, norm_sections)
        # Geo extraction (best-effort)
        company_name = self._extract_company_name(domain, norm_sections, jsonld)
        city = self._extract_city(norm_body, norm_sections, jsonld)
        country = self._extract_country(norm_body, jsonld, domain)

        # Signals dict (debug)
        signals: Dict[str, float] = {}
        reasons: List[str] = []

        # KSA signal
        ksa_signal = self._ksa_signal(norm_body, contacts, domain, city_hint=city)
        non_ksa = self._non_ksa_signal(norm_body, ksa_signal, domain=domain)

        signals["ksa_signal"] = float(ksa_signal)
        signals["non_ksa"] = 1.0 if non_ksa else 0.0
        signals["source_type"] = float(
            {
                "company_site": 1,
                "ecommerce": 0.8,
                "retail_catalog": 0.6,
                "marketplace": 0.5,
                "article": 0.2,
                "gov_edu": 0.0,
            }.get(source_type, 0.3)  # 0.3 as default
        )

        # Contacts signals (small confirmation, not a reason to be relevant)
        has_email = 1.0 if contacts.get("emails") else 0.0
        has_phone = 1.0 if contacts.get("phones") else 0.0
        has_whatsapp = 1.0 if contacts.get("whatsapp") else 0.0

        signals["has_email"] = has_email
        signals["has_phone"] = has_phone
        signals["has_whatsapp"] = has_whatsapp

        # --- Weighted term counting across sections ---
        # Section weights: title/meta strong, H1/H2 strong, contact/footer medium, body normal
        section_weights = {
            "title": 3.0,
            "meta": 3.0,
            "h1": 2.0,
            "h2": 1.5,
            "contact": 2.0,
            "footer": 1.5,
            "body": 1.0,
        }

        # Build a combined mapping used by the counter
        text_map = dict(norm_sections)
        text_map["body"] = norm_body

        hp_terms = self.keywords.get("high_precision", [])
        mp_terms = self.keywords.get("mid_precision", [])
        biz_terms = self.keywords.get("business_intent", [])
        ind_terms = self.keywords.get("individual_markers", [])  # existing
        bus_markers = self.keywords.get("business_markers", [])  # existing

        neg_terms = self.negative_keywords

        # Weighted counts (continuous, not binary)
        hp = self._count_terms_weighted(text_map, hp_terms, section_weights)
        mp = self._count_terms_weighted(text_map, mp_terms, section_weights)
        biz = self._count_terms_weighted(text_map, biz_terms, section_weights)
        neg = self._count_terms_weighted(text_map, neg_terms, section_weights)

        # Also keep some legacy bucket signals for transparency/debug
        kitchen_terms = self.keywords.get("kitchen", [])
        kitchen_signal = self._bucket_signal(norm_body, kitchen_terms, min_hits=1)
        interior_signal = self._bucket_signal(norm_body, self.keywords.get("interior", []), min_hits=1)
        fitout_signal = self._bucket_signal(norm_body, self.keywords.get("fitout", []), min_hits=2)
        architect_signal = self._bucket_signal(norm_body, self.keywords.get("architect", []), min_hits=1)
        portfolio_signal = self._bucket_signal(norm_body, self.keywords.get("portfolio", []), min_hits=2)
        kitchen_hits = self._count_terms_weighted(text_map, kitchen_terms, section_weights)

        signals["kitchen_signal"] = float(kitchen_signal)
        signals["kitchen_hits"] = float(kitchen_hits)
        signals["interior_signal"] = float(interior_signal)
        signals["fitout_signal"] = float(fitout_signal)
        signals["architect_signal"] = float(architect_signal)
        signals["portfolio_signal"] = float(portfolio_signal)

        # Text size normalization
        total_words = self._word_count(norm_body)
        signals["total_words"] = float(total_words)
        empty_content = total_words <= 30 or len(norm_body.strip()) < 200
        signals["empty_content"] = 1.0 if empty_content else 0.0

        # Density (anti "big site" inflation)
        # Use a floor of 300 words to avoid short pages exploding.
        denom = max(total_words, 300)
        density = (hp + 0.5 * mp) / float(denom)
        signals["density"] = float(density)

        # Lead type logic (per your clarified requirement):
        # - B2C: mostly marketplace/listing + strong individual intent + weak business markers
        # - B2B: company sites, ecommerce, etc.
        individual_intent = self._count_terms_weighted(text_map, ind_terms, section_weights)
        business_marker_strength = self._count_terms_weighted(text_map, bus_markers, section_weights)

        service_terms = [
            "fit-out", "fit out", "fitout", "interior fit-out", "interior finishing",
            "contracting", "contractor", "general contractor", "turnkey",
            "engineering", "engineering consultancy", "engineering office",
            "architect", "architectural design", "design office", "interior works",
            "تشطيب", "مقاولات", "مقاول", "تسليم مفتاح", "فيت اوت",
            "مكتب هندسي", "استشارات هندسية", "تصميم معماري",
            "مقاول عام", "مقاولات عامة", "مقاولات بناء", "مقاولات إنشائية",
            "تصميم داخلي", "تصميم داخلي معماري", "ديكور", "أعمال ديكور",
            "تشطيبات", "تشطيبات داخلية", "أعمال تشطيب", "تنفيذ داخلي",
            "تجهيز داخلي", "اعمال داخلية", "أعمال داخلية", "أعمال معمارية",
            "إشراف هندسي", "إدارة مشاريع", "إدارة مشروع", "إدارة تنفيذ",
            "مكتب استشاري", "استشاري هندسي", "مكتب تصميم", "تصميم هندسي",
            "اعمال انشائية", "أعمال إنشائية", "مقاول بناء",
        ]
        service_evidence = self._count_terms_weighted(text_map, service_terms, section_weights)
        developer_terms = [
            # EN: developers / contractors / consultancies
            "real estate developer", "property developer", "master developer",
            "real estate development", "property development", "development company",
            "general contractor", "construction company", "contracting company",
            "engineering consultancy", "engineering office", "design & build", "design-build",
            "project management", "pmo",
            # AR
            "مطوّر عقاري", "مطور عقاري", "تطوير عقاري", "شركة تطوير", "شركة تطوير عقاري",
            "مقاول عام", "مقاولات عامة", "شركة مقاولات",
            "مكتب هندسي", "استشارات هندسية",
            "إدارة مشاريع", "ادارة مشاريع", "إدارة المشروع", "ادارة المشروع",
            "تطوير مشاريع",
        ]
        developer_evidence = self._count_terms_weighted(text_map, developer_terms, section_weights)

        if source_type == "marketplace" and individual_intent >= 2.0 and business_marker_strength < 2.0:
            lead_type = "B2C"
        else:
            lead_type = "B2B"

        signals["individual_intent"] = float(individual_intent)
        signals["business_markers_strength"] = float(business_marker_strength)
        signals["service_evidence"] = float(service_evidence)
        signals["developer_evidence"] = float(developer_evidence)

        # Override ecommerce when strong B2B/portfolio evidence exists
        b2b_override_terms = [
            "projects", "our projects", "case studies", "clients", "portfolio",
            "scope of work", "services", "fit-out", "turnkey",
            "مشاريع", "مشاريعنا", "دراسات حالة", "عملاء", "بورتفوليو",
            "نطاق العمل", "الخدمات", "الخدمات المقدمة", "تشطيب", "تسليم مفتاح",
        ]
        b2b_override_hits = self._count_terms_weighted(text_map, b2b_override_terms, section_weights)
        b2b_evidence = (
            portfolio_signal == 1
            or fitout_signal == 1
            or (hp >= 6.0)
            or (b2b_override_hits >= 1.0)
        )
        if source_type == "ecommerce" and b2b_evidence:
            source_type = "company_site"
            signals["ecommerce_overridden"] = 1.0
            signals["retail_catalog"] = 0.0
        elif source_type == "ecommerce":
            source_type = "retail_catalog"
            signals["ecommerce_overridden"] = 0.0
            signals["retail_catalog"] = 1.0
        else:
            signals["ecommerce_overridden"] = 0.0
            signals["retail_catalog"] = 0.0
        signals["source_type"] = float(
            {
                "company_site": 1,
                "ecommerce": 0.8,
                "retail_catalog": 0.6,
                "marketplace": 0.5,
                "article": 0.2,
                "gov_edu": 0.0,
            }.get(source_type, 0.3)
        )

        # Page-type penalty
        page_penalty = 0
        if source_type == "gov_edu":
            page_penalty += self.weights["penalty_gov_edu"]
        elif source_type == "article":
            page_penalty += self.weights["penalty_article"]
        elif source_type == "marketplace":
            page_penalty += self.weights["penalty_marketplace"]
        elif source_type == "ecommerce":
            page_penalty += self.weights["penalty_ecommerce"]
        elif source_type == "retail_catalog":
            page_penalty += self.weights["penalty_retail_catalog"]

        signals["page_penalty"] = float(page_penalty)

        # Contacts bonus (capped)
        contact_bonus = (
            int(has_email) * self.weights["has_email"] +
            int(has_phone) * self.weights["has_phone"] +
            int(has_whatsapp) * self.weights["has_whatsapp"]
        )
        contact_bonus = min(contact_bonus, 10)  # cannot be bigger than 10
        signals["contact_bonus"] = float(contact_bonus)

        # Geo bonus/penalty
        geo_bonus = 0
        if ksa_signal >= 0.7:
            geo_bonus += self.weights["ksa_bonus"]
        if non_ksa:
            geo_bonus += self.weights["non_ksa_penalty"]
        signals["geo_bonus"] = float(geo_bonus)

        # --- Core score components (log-normalized) ---
        # log_norm(x, cap): maps 0..cap+ to 0..1
        hp_part = self.weights["hp_max"] * self._log_norm(hp, cap=12.0)
        mp_part = self.weights["mp_max"] * self._log_norm(mp, cap=40.0)
        biz_part = self.weights["biz_max"] * self._log_norm(biz, cap=12.0)

        # Density becomes decisive (scaled)
        density_part = self.weights["density_max"] * min(1.0, density * 120.0)

        # Negative penalty always applies (and is stronger if negative appears in title/h1)
        neg_title_h1 = self._count_terms_weighted(
            {"title": norm_sections.get("title", ""), "h1": norm_sections.get("h1", ""), "h2": norm_sections.get("h2", "")},
            neg_terms,
            {"title": 3.0, "h1": 2.0, "h2": 1.5},
        )
        hard_neg_title_h1 = self._count_terms_weighted(
            {"title": norm_sections.get("title", ""), "h1": norm_sections.get("h1", "")},
            self.hard_negative_terms,
            {"title": 3.0, "h1": 2.0},
        )
        hard_neg_domain = 1.0 if self._count_terms_weighted({"d": domain}, self.hard_negative_terms, {"d": 1.0}) > 0 else 0.0
        hard_negative_hit = (neg_title_h1 > 0.0) or (hard_neg_title_h1 > 0.0) or (hard_neg_domain > 0.0)

        neg_part = self.weights["negative_max"] * self._log_norm(neg + 1.5 * neg_title_h1, cap=8.0)

        signals["hp_count"] = float(hp)
        signals["mp_count"] = float(mp)
        signals["biz_count"] = float(biz)
        signals["neg_count"] = float(neg)
        signals["neg_title_h1"] = float(neg_title_h1)
        signals["hard_negative_hit"] = 1.0 if hard_negative_hit else 0.0
        signals["article_guard_blocked"] = 1.0 if article_guard_blocked else 0.0

        off_vertical_terms = [
            "water leak", "leak detection", "pest control", "cleaning services",
            "car", "auto", "dealership", "bank", "loan", "insurance",
            "travel", "hotel", "flight", "food delivery",
            "real estate listings", "property listings", "classifieds",
            "تسرب", "تسربات", "كشف تسرب", "مكافحة حشرات", "تنظيف",
            "سيارات", "معرض سيارات", "بنك", "قرض", "تأمين",
            "سفر", "حجز", "توصيل الطعام", "طلبات الطعام", "عقار", "إعلانات",
        ]
        off_vertical_score = self._count_terms_weighted(
            {
                "domain": domain,
                "title": norm_sections.get("title", ""),
                "h1": norm_sections.get("h1", ""),
                "body": norm_body,
            },
            off_vertical_terms,
            {"domain": 3.0, "title": 2.5, "h1": 2.0, "body": 1.0},
        )
        off_vertical_weak_target = (hp < 2.0 and fitout_signal == 0 and architect_signal == 0 and interior_signal == 0)
        off_vertical_guard = off_vertical_score >= 2.0 and off_vertical_weak_target
        sanity_penalty = -35.0 if off_vertical_guard else 0.0
        signals["off_vertical_guard"] = 1.0 if off_vertical_guard else 0.0
        signals["sanity_penalty"] = float(sanity_penalty)

        # --- Gate conditions (prevents random one-off mentions) ---
        # Require KSA or explicit KSA contact/domain (unless you want multi-country)
        ksa_gate = (ksa_signal >= 0.7) or domain.endswith(".sa") or any(p.startswith("+966") for p in contacts.get("phones", []))
        # Require evidence of service/business, not just one word
        content_gate = (hp >= 2.0) or (hp >= 1.0 and biz >= 2.0 and density >= 0.015)
        # Frequency-based gate (derived from seed vs negative analysis)
        keyword_gate = (hp >= 1.0) or (mp >= 20.0 and kitchen_hits >= 10.0)
        # Service override: strong fit-out/contracting/engineering + KSA + contacts
        service_gate = (
            ksa_gate
            and (has_email == 1.0 or has_phone == 1.0)
            and (fitout_signal == 1 or architect_signal == 1 or interior_signal == 1 or service_evidence >= 3.0)
            and (biz >= 2.0 or hp >= 1.0)
        )
        developer_gate = (
            ksa_gate
            and (developer_evidence >= 2.0)
            and (biz >= 2.0 or mp >= 20.0)
        )
        # Fallback gate: kitchen/fit-out evidence even with low density
        fallback_gate = (
            ksa_gate
            and (has_email == 1.0 or has_phone == 1.0)
            and (mp >= 20.0)
            and (
                (hp >= 2.0)
                or (fitout_signal == 1)
                or (kitchen_signal == 1 and portfolio_signal == 1)
            )
            and (neg < 40.0)
        )
        if keyword_gate and ksa_gate:
            content_gate = True
        if developer_gate:
            content_gate = True
        if fallback_gate or service_gate:
            content_gate = True
        if empty_content:
            content_gate = False

        signals["ksa_gate"] = 1.0 if ksa_gate else 0.0
        signals["content_gate"] = 1.0 if content_gate else 0.0
        signals["content_gate_fallback"] = 1.0 if fallback_gate else 0.0
        signals["content_gate_service"] = 1.0 if service_gate else 0.0
        signals["content_gate_keyword"] = 1.0 if keyword_gate else 0.0
        signals["content_gate_developer"] = 1.0 if developer_gate else 0.0

        # Cap negative penalty when strong target evidence exists (unless hard-negative evidence)
        strong_positive = (
            hp >= 6.0
            or (fitout_signal == 1 and mp >= 20.0)
            or (kitchen_signal == 1 and portfolio_signal == 1 and mp >= 20.0 and neg < 20.0)
        )
        if strong_positive and not hard_negative_hit:
            neg_part = max(neg_part, -15.0)
            signals["neg_capped"] = 1.0
        else:
            signals["neg_capped"] = 0.0

        base_score = (
            hp_part
            + mp_part
            + biz_part
            + density_part
            + float(geo_bonus)
            + float(contact_bonus)
            + float(page_penalty)
            + neg_part
            + float(sanity_penalty)
        )

        # Showroom bonus for kitchen-focused companies with contacts and low negatives
        showroom_bonus = 0.0
        if source_type in ("company_site", "ecommerce") and lead_type == "B2B":
            bonus_category = self._categorize_business(norm_body, norm_sections, contacts, lead_type=lead_type)
            if (
                bonus_category == "showroom"
                and kitchen_signal == 1
                and (has_email == 1.0 or has_phone == 1.0)
                and neg <= 5.0
            ):
                showroom_bonus = 10.0
        if showroom_bonus:
            base_score += showroom_bonus
        signals["showroom_bonus"] = float(showroom_bonus)

        # If gates fail, cap score so they don't accidentally pass
        if not ksa_gate:
            base_score = min(base_score, 55.0)
        if not content_gate:
            base_score = min(base_score, 55.0)

        # Clamp to 0..100 (int)
        score = int(max(0, min(100, round(base_score))))

        # Decide relevant + confidence
        relevant, confidence = self._decision(score)
        decision = "accept" if score >= self.thresholds["relevant"] else ("review" if score >= self.thresholds["maybe_low"] else "reject")
        if empty_content:
            decision = "review"
        if developer_gate and decision == "reject":
            decision = "review"
        if empty_content:
            relevant = False
            confidence = min(confidence, 0.35)
        decision, relevant, confidence = self._binary_decision(decision, relevant, confidence)

        # Categorize only when it makes sense
        category = "unknown"
        if score >= self.thresholds["maybe_low"] and source_type in ("company_site", "ecommerce"):
            category = self._categorize_business(norm_body, norm_sections, contacts, lead_type=lead_type)
        elif lead_type == "B2C" and source_type == "marketplace":
            category = "individual"
        if decision != "accept":
            category = "unknown"

        if self.mode == "recall":
            (
                score,
                decision,
                relevant,
                confidence,
                recall_signals,
                recall_reasons,
            ) = self._decision_recall(
                domain=domain,
                source_type=source_type,
                ksa_signal=ksa_signal,
                non_ksa=non_ksa,
                contacts=contacts,
                empty_content=empty_content,
                kitchen_signal=kitchen_signal,
                kitchen_hits=kitchen_hits,
                fitout_signal=fitout_signal,
                interior_signal=interior_signal,
                architect_signal=architect_signal,
                hard_negative_hit=hard_negative_hit,
            )
            signals.update(recall_signals)

            category = "unknown"
            if decision in ("accept", "review") and source_type in ("company_site", "ecommerce", "retail_catalog"):
                category = self._categorize_business(norm_body, norm_sections, contacts, lead_type=lead_type)
            elif lead_type == "B2C" and source_type == "marketplace":
                category = "individual"

            decision, relevant, confidence = self._binary_decision(decision, relevant, confidence)
            if decision != "accept":
                category = "unknown"

            reasons = self._build_recall_reasons(
                decision=decision,
                source_type=source_type,
                category=category,
                lead_type=lead_type,
                company_name=company_name,
                city=city,
                country=country,
                contacts=contacts,
                non_ksa=non_ksa,
                ksa_signal=ksa_signal,
                empty_content=empty_content,
                hard_negative_hit=hard_negative_hit,
                kitchen_signal=kitchen_signal,
                fitout_signal=fitout_signal,
                interior_signal=interior_signal,
                architect_signal=architect_signal,
                recall_signals=recall_signals,
            )

            return SiteEvaluation(
                url=url,
                domain=domain,
                relevant=relevant,
                relevance_score=score,
                confidence=confidence,
                decision=decision,
                lead_type=lead_type,
                category=category,
                contacts=contacts,
                signals=signals,
                reasons=reasons,
                source_type=source_type,
                company_name=company_name,
                city=city,
                country=country,
                empty_content=bool(empty_content),
            )

        # Reasons
        reasons.extend(self._build_reasons(
            signals=signals,
            contacts=contacts,
            score=score,
            relevant=relevant,
            category=category,
            lead_type=lead_type,
            source_type=source_type,
            company_name=company_name,
            city=city,
            country=country,
        ))

        return SiteEvaluation(
            url=url,
            domain=domain,
            relevant=relevant,
            relevance_score=score,
            confidence=confidence,
            decision=decision,
            lead_type=lead_type,
            category=category,
            contacts=contacts,
            signals=signals,
            reasons=reasons,
            source_type=source_type,
            company_name=company_name,
            city=city,
            country=country,
            empty_content=bool(empty_content),
        )

    def evaluate_url(self, url: str, timeout: float = 15.0) -> SiteEvaluation:
        html = self._fetch_html(url, timeout=timeout)
        return self.evaluate(url, html=html)

    # -----------------------------
    # Decision & Reasons
    # -----------------------------

    def _decision(self, score: int) -> Tuple[bool, float]:
        if score >= self.thresholds["relevant"]:
            # confidence grows with margin
            conf = 0.72 + (score - self.thresholds["relevant"]) / 100.0
            return True, float(min(0.95, max(0.0, conf)))
        if score < self.thresholds["maybe_low"]:
            conf = 0.60 + (self.thresholds["maybe_low"] - score) / 100.0
            return False, float(min(0.90, max(0.0, conf)))
        return False, 0.55

    def _binary_decision(self, decision: str, relevant: bool, confidence: float) -> Tuple[str, bool, float]:
        """
        Collapse decision to two categories: accept or reject.
        """
        if decision != "accept":
            return "reject", False, float(min(confidence, 0.60))
        return "accept", True, confidence

    def _decision_recall(
        self,
        domain: str,
        source_type: str,
        ksa_signal: float,
        non_ksa: bool,
        contacts: Dict[str, List[str]],
        empty_content: bool,
        kitchen_signal: float,
        kitchen_hits: float,
        fitout_signal: float,
        interior_signal: float,
        architect_signal: float,
        hard_negative_hit: bool,
    ) -> Tuple[int, str, bool, float, Dict[str, float], List[str]]:
        """
        Recall-first decision: reject only when strong evidence suggests a clearly irrelevant site.
        Everything else is accept (or review if content is too thin).
        """
        phones = contacts.get("phones") or []
        ksa_soft = (
            ksa_signal >= 0.4
            or domain.endswith(".sa")
            or any(p.startswith("+966") for p in phones)
        )
        non_ksa_strong = bool(non_ksa) and not ksa_soft

        clear_kitchen = (
            kitchen_signal == 1
            or kitchen_hits >= 1.0
            or fitout_signal == 1
            or interior_signal == 1
            or architect_signal == 1
        )
        hard_negative_strong = bool(hard_negative_hit) and not clear_kitchen
        source_reject = source_type in ("gov_edu", "marketplace")

        recall_signals = {
            "recall_mode": 1.0,
            "recall_non_ksa_reject": 1.0 if non_ksa_strong else 0.0,
            "recall_source_reject": 1.0 if source_reject else 0.0,
            "recall_hard_negative_reject": 1.0 if hard_negative_strong else 0.0,
            "recall_clear_kitchen": 1.0 if clear_kitchen else 0.0,
            "recall_empty_review": 1.0 if empty_content else 0.0,
        }

        if non_ksa_strong or source_reject or hard_negative_strong:
            return 10, "reject", False, 0.80, recall_signals, [
                "Recall mode: rejected due to strong non-KSA or off-vertical evidence."
            ]
        if empty_content:
            return 50, "review", False, 0.35, recall_signals, [
                "Recall mode: insufficient content; keep for manual review."
            ]
        return 80, "accept", True, 0.60, recall_signals, [
            "Recall mode: accepted unless clearly irrelevant."
        ]

    def _build_recall_reasons(
        self,
        decision: str,
        source_type: str,
        category: str,
        lead_type: str,
        company_name: str,
        city: str,
        country: str,
        contacts: Dict[str, List[str]],
        non_ksa: bool,
        ksa_signal: float,
        empty_content: bool,
        hard_negative_hit: bool,
        kitchen_signal: float,
        fitout_signal: float,
        interior_signal: float,
        architect_signal: float,
        recall_signals: Dict[str, float],
    ) -> List[str]:
        r: List[str] = []
        r.append(
            f"Recall mode: decision={decision}; source_type={source_type}; category={category}; "
            f"lead_type={lead_type}; name={company_name}; city={city}; country={country}"
        )
        r.append(
            f"Signals: ksa_signal={ksa_signal:.2f}, non_ksa={int(non_ksa)}, empty_content={int(empty_content)}, "
            f"hard_negative_hit={int(hard_negative_hit)}, kitchen={int(kitchen_signal)}, "
            f"fitout={int(fitout_signal)}, interior={int(interior_signal)}, architect={int(architect_signal)}"
        )
        if contacts.get("emails"):
            r.append(f"Found emails: {contacts['emails'][:3]}")
        if contacts.get("phones"):
            r.append(f"Found phones: {contacts['phones'][:3]}")
        if contacts.get("whatsapp"):
            r.append("Found WhatsApp link/number")
        if recall_signals.get("recall_non_ksa_reject", 0) >= 1:
            r.append("Rejected: strong non-KSA evidence.")
        if recall_signals.get("recall_source_reject", 0) >= 1:
            r.append("Rejected: gov/edu or marketplace source type.")
        if recall_signals.get("recall_hard_negative_reject", 0) >= 1:
            r.append("Rejected: hard-negative terms without kitchen evidence.")
        if recall_signals.get("recall_empty_review", 0) >= 1:
            r.append("Review: empty/insufficient content.")
        return r

    def _build_reasons(
        self,
        signals: Dict[str, float],
        contacts: Dict[str, List[str]],
        score: int,
        relevant: bool,
        category: str,
        lead_type: str,
        source_type: str,
        company_name: str,
        city: str,
        country: str,
    ) -> List[str]:
        r: List[str] = []
        r.append(
            f"Score={score} => {'RELEVANT' if relevant else 'NOT SURE/NOT RELEVANT'}; "
            f"source_type={source_type}; category={category}; lead_type={lead_type}; "
            f"name={company_name}; city={city}; country={country}"
        )

        # Contacts
        if contacts.get("emails"):
            r.append(f"Found emails: {contacts['emails'][:3]}")
        if contacts.get("phones"):
            r.append(f"Found phones: {contacts['phones'][:3]}")
        if contacts.get("whatsapp"):
            r.append("Found WhatsApp link/number")

        # Gates
        if signals.get("ksa_gate", 0) < 1:
            r.append("KSA gate not satisfied (weak KSA evidence) -> score capped")
        if signals.get("content_gate", 0) < 1:
            r.append("Content gate not satisfied (insufficient high-precision/service evidence) -> score capped")
        if signals.get("content_gate_fallback", 0) >= 1:
            r.append("Content gate passed via fallback (mp/biz/contacts/KSA)")
        if signals.get("content_gate_service", 0) >= 1:
            r.append("Content gate passed via service override (fit-out/contracting/engineering + KSA + contacts)")
        if signals.get("content_gate_keyword", 0) >= 1:
            r.append("Content gate passed via keyword frequency (hp/mp+kitchen)")
        if signals.get("content_gate_developer", 0) >= 1:
            r.append("Content gate passed via developer/contractor evidence")
        if signals.get("empty_content", 0) >= 1:
            r.append("Empty/blocked/JS-only content: insufficient text extracted")

        # Geo
        if signals.get("ksa_signal", 0) >= 0.7:
            r.append("KSA signal detected (city/ksa/saudi/+966/.sa)")
        if signals.get("non_ksa", 0) > 0:
            r.append("Strong non-KSA signal detected -> penalty applied")

        # Page type
        if source_type in ("gov_edu", "article", "marketplace", "ecommerce", "retail_catalog"):
            r.append(f"Page type classified as '{source_type}' -> penalty/handling applied")
        if signals.get("article_guard_blocked", 0) >= 1:
            r.append("Article classification blocked by homepage URL guard")
        if signals.get("ecommerce_overridden", 0) >= 1:
            r.append("Ecommerce classification overridden to company_site due to B2B/portfolio evidence")
        if signals.get("off_vertical_guard", 0) >= 1:
            r.append("Off-vertical business signals detected; applied sanity penalty (needs review)")

        # Core counters
        r.append(
            "Counts: "
            f"hp={signals.get('hp_count', 0):.1f}, "
            f"mp={signals.get('mp_count', 0):.1f}, "
            f"biz={signals.get('biz_count', 0):.1f}, "
            f"neg={signals.get('neg_count', 0):.1f}, "
            f"density={signals.get('density', 0):.4f}, "
            f"words={int(signals.get('total_words', 0))}"
        )

        if signals.get("neg_title_h1", 0) > 0:
            r.append("Negative keywords detected in title/H1 -> stronger penalty")
        if signals.get("neg_capped", 0) >= 1:
            r.append("Negative penalty capped due to strong positive signals")
        if signals.get("showroom_bonus", 0) > 0:
            r.append("Showroom bonus applied (kitchen + contacts + low negatives)")

        return r

    # -----------------------------
    # Categorization
    # -----------------------------

    def _categorize_business(
        self,
        norm_body: str,
        norm_sections: Dict[str, str],
        contacts: Dict[str, List[str]],
        lead_type: str,
    ) -> str:
        """
        Category for business sites only (B2B mostly).
        Uses weighted evidence in strong sections to avoid random mentions.
        """

        if lead_type == "B2C":
            return "individual"

        # If site looks like aggregator/news even after source_type, stay cautious
        text_map = {
            "title": norm_sections.get("title", ""),
            "meta": norm_sections.get("meta", ""),
            "h1": norm_sections.get("h1", ""),
            "h2": norm_sections.get("h2", ""),
            "contact": norm_sections.get("contact", ""),
            "footer": norm_sections.get("footer", ""),
            "body": norm_body,
        }
        section_weights = {
            "title": 3.0,
            "meta": 3.0,
            "h1": 2.0,
            "h2": 1.5,
            "contact": 2.0,
            "footer": 1.5,
            "body": 1.0,
        }

        # Bucket scores (weighted counts, then squashed)
        def bucket_score(bucket_name: str, cap: float, boost: float = 1.0) -> float:
            terms = self.keywords.get(bucket_name, [])
            c = self._count_terms_weighted(text_map, terms, section_weights)
            return boost * self._log_norm(c, cap=cap)

        showroom = bucket_score("showroom", cap=10.0, boost=1.05)
        fitout = bucket_score("fitout", cap=14.0, boost=1.00)
        architect = bucket_score("architect", cap=14.0, boost=1.02)
        developer = bucket_score("developer", cap=10.0, boost=0.95)
        designer = bucket_score("designer", cap=10.0, boost=1.00)

        # Extra heuristics: showroom likes address/branches/opening hours
        showroom_proof_terms = ["open", "hours", "opening hours", "visit", "branches", "branch", "showroom", "معرض", "فروع", "ساعات العمل", "زوروا المعرض"]
        showroom_proof = self._count_terms_weighted(text_map, showroom_proof_terms, section_weights)
        if showroom_proof >= 2:
            showroom = min(1.0, showroom + 0.10)

        # Architect vs fit-out tie-break: architect wins if permits/supervision are present
        permit_terms = ["permit", "building permit", "site supervision", "engineering office", "رخصة", "تصريح", "إشراف", "مكتب هندسي"]
        permit_sig = self._count_terms_weighted(text_map, permit_terms, section_weights)

        # Decide
        scores = {
            "showroom": showroom,
            "fit-out": fitout,
            "architect": architect,
            "developer": developer,
            "designer": designer,
        }

        top_val = max(scores.values()) if scores else 0.0
        if top_val <= 0.0:
            return "unknown"

        # Priority order for ties (slightly prefer architect when equal and permit signals exist)
        # Default: architect > fit-out > showroom > designer > developer
        priority = ["architect", "fit-out", "showroom", "designer", "developer"]

        # If permit signals exist, prioritize architect more strongly
        if permit_sig >= 1.0:
            priority = ["architect", "fit-out", "showroom", "designer", "developer"]

        for cat in priority:
            if abs(scores.get(cat, 0.0) - top_val) < 1e-9:
                return cat

        return "unknown"

    # -----------------------------
    # Source type classification
    # -----------------------------

    def _classify_source_type(self, url: str, domain: str, norm_body: str, norm_sections: Dict[str, str]) -> Tuple[str, bool]:
        path = urlparse(url).path.lower()
        segments = [s for s in path.split("/") if s]
        homepage_like = path in ("", "/", "/en", "/en/", "/ar", "/ar/") or len(segments) <= 1
        article_guard_blocked = False
        article_patterns = ["/blog", "/news", "/post", "/article", "/insights", "/press", "/updates"]

        # GOV/EDU (hard non-lead in most cases)
        if domain.endswith(".gov.sa") or domain.endswith(".edu.sa"):
            return "gov_edu", article_guard_blocked
        if any(x in domain for x in ["moc.gov.sa", "momah.gov.sa", "balady.gov.sa"]) or "e-services" in norm_body:
            return "gov_edu", article_guard_blocked

        # Marketplace / directories
        marketplace_hosts = [
            "opensooq", "haraj", "aqar", "bayut", "linkedin", "amazon", "ikea", "tamimimarkets",
            "noon", "dubizzle", "olx",
        ]
        if any(h in domain for h in marketplace_hosts):
            return "marketplace", article_guard_blocked
        if any(x in path for x in ["/tags/", "/property/", "/details", "/listing", "/classified", "/ad/", "/in/"]):
            return "marketplace", article_guard_blocked

        # Article/blog/news
        if any(x in path for x in article_patterns):
            if homepage_like:
                article_guard_blocked = True
            else:
                return "article", article_guard_blocked
        # Heuristic: "related articles", "read more", author/date patterns
        if any(x in path for x in article_patterns) and (
            ("read more" in norm_body) or ("related articles" in norm_body) or re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", norm_body)
        ):
            if homepage_like:
                article_guard_blocked = True
            else:
                # Only classify as article if service-proof is weak in title/h1
                title_h1 = (norm_sections.get("title", "") + " " + norm_sections.get("h1", "")).strip()
                service_terms = self.keywords.get("high_precision", []) + self.keywords.get("business_intent", [])
                if self._count_terms_weighted({"t": title_h1}, service_terms, {"t": 1.0}) < 2.0:
                    return "article", article_guard_blocked

        # Ecommerce
        if any(x in path for x in ["/product", "/category", "/categories", "/catalog", "/cart", "/checkout", "/shop"]):
            return "ecommerce", article_guard_blocked
        if any(x in norm_body for x in ["add to cart", "checkout", "basket", "سلة", "الدفع"]):
            return "ecommerce", article_guard_blocked

        return "company_site", article_guard_blocked

    # -----------------------------
    # Contacts & parsing helpers
    # -----------------------------

    def _extract_contacts(self, text: str, url: str, html: Optional[str] = None) -> Dict[str, List[str]]:
        emails = set(self._find_emails(text))
        phones = set(self._find_phones(text))
        whatsapp = set(self._find_whatsapp(text))

        if html:
            emails |= set(self._find_emails_in_html_attrs(html))
            phones |= set(self._find_phones_in_html_attrs(html))
            whatsapp |= set(self._find_whatsapp_in_html_attrs(html))

            # Cloudflare email protection (data-cfemail)
            emails |= set(self._find_cfemail(html))

            # Simple JS obfuscations like "info" + "@" + "domain.com"
            emails |= set(self._find_js_concat_emails(html))

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
        cleaned = (
            text.replace("(at)", "@")
            .replace("[at]", "@")
            .replace(" at ", "@")
            .replace(" (at) ", "@")
            .replace(" [at] ", "@")
        )
        pattern = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
        return [m.group(0).lower() for m in pattern.finditer(cleaned)]

    def _find_emails_in_html_attrs(self, html: str) -> List[str]:
        raw = html_lib.unescape(html)
        raw = unquote(raw)
        emails: set[str] = set()

        # mailto:
        for m in re.finditer(r'(?i)\bmailto:\s*([^"\'\s>]+)', raw):
            chunk = m.group(1).split("?")[0]
            for part in re.split(r"[,\s;]+", chunk):
                part = part.strip()
                if part:
                    emails.update(self._find_emails(part))

        # common attrs that may store emails
        attr_vals = re.findall(
            r'(?is)\b(?:href|data-email|data-mail|data-contact|data-clipboard-text)\s*=\s*["\']([^"\']+)["\']',
            raw,
        )
        for v in attr_vals:
            emails.update(self._find_emails(v))

        return sorted(emails)

    def _find_phones(self, text: str) -> List[str]:
        candidates = re.findall(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
        out = set()

        for raw in candidates:
            raw = raw.strip()
            try:
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

        # tel:
        for m in re.finditer(r'(?i)\btel:\s*([^"\'\s>]+)', raw):
            chunk = m.group(1).split("?")[0]
            phones.update(self._find_phones(chunk))

        # typical attrs
        attr_vals = re.findall(r'(?is)\b(?:href|data-phone|data-tel|data-contact)\s*=\s*["\']([^"\']+)["\']', raw)
        for v in attr_vals:
            phones.update(self._find_phones(v))

        return sorted(phones)

    def _find_whatsapp(self, text: str) -> List[str]:
        pattern = re.compile(r"(https?://(?:wa\.me|api\.whatsapp\.com)/[^\s\"'<]+)", re.IGNORECASE)
        return [m.group(1) for m in pattern.finditer(text)]

    def _find_whatsapp_in_html_attrs(self, html: str) -> List[str]:
        raw = html_lib.unescape(html)
        raw = unquote(raw)

        wa: set[str] = set()
        wa.update(self._find_whatsapp(raw))

        hrefs = re.findall(r'(?is)\bhref\s*=\s*["\']([^"\']+)["\']', raw)
        for h in hrefs:
            wa.update(self._find_whatsapp(h))

        return sorted(wa)

    def _find_cfemail(self, html: str) -> List[str]:
        """
        Decode Cloudflare email obfuscation: data-cfemail="hex..."
        """
        raw = html_lib.unescape(html)
        emails: set[str] = set()
        for m in re.finditer(r'(?is)\bdata-cfemail\s*=\s*["\']([0-9a-fA-F]+)["\']', raw):
            hexstr = m.group(1).strip()
            try:
                data = bytes.fromhex(hexstr)
                if not data:
                    continue
                key = data[0]
                decoded = bytes(b ^ key for b in data[1:]).decode("utf-8", errors="ignore")
                # validate
                for e in self._find_emails(decoded):
                    emails.add(e)
            except Exception:
                continue
        return sorted(emails)

    def _find_js_concat_emails(self, html: str) -> List[str]:
        """
        Catch simple patterns like "info" + "@" + "domain.com"
        """
        raw = html_lib.unescape(html)
        raw = unquote(raw)
        emails: set[str] = set()

        # Very simple: "something" + "@" + "something"
        for m in re.finditer(
            r'(?is)(["\'])([a-zA-Z0-9._%+\-]{1,64})\1\s*\+\s*(["\'])@\3\s*\+\s*(["\'])([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\4',
            raw,
        ):
            local = m.group(2)
            dom = m.group(5)
            candidate = f"{local}@{dom}".lower()
            if self._find_emails(candidate):
                emails.add(candidate)

        return sorted(emails)

    # -----------------------------
    # HTML / JSON-LD extraction
    # -----------------------------

    def _html_to_text(self, html: str) -> str:
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        html = re.sub(r"(?s)<[^>]+>", " ", html)
        html = html_lib.unescape(html)
        html = html.replace("\xa0", " ")
        return re.sub(r"\s+", " ", html).strip()

    def _extract_sections(self, html: str) -> Dict[str, str]:
        """
        Lightweight section extractor. (No BeautifulSoup dependency.)
        """
        raw = html

        def _first(pattern: str) -> str:
            m = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
            if not m:
                return ""
            return self._strip_tags(m.group(1))

        title = _first(r"<title[^>]*>(.*?)</title>")
        meta = _first(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']')
        og_site = _first(r'<meta[^>]+property=["\']og:site_name["\'][^>]+content=["\'](.*?)["\']')
        og_title = _first(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']')
        h1 = self._strip_tags(" ".join(re.findall(r"(?is)<h1[^>]*>(.*?)</h1>", raw)[:1]))
        h2 = self._strip_tags(" ".join(re.findall(r"(?is)<h2[^>]*>(.*?)</h2>", raw)[:2]))

        # footer/contact (very heuristic)
        footer_html = " ".join(re.findall(r"(?is)<footer[^>]*>(.*?)</footer>", raw)[:1])
        footer = self._strip_tags(footer_html)

        # contact-ish blocks: look for "contact" sections by id/class
        contact_blocks = []
        for m in re.finditer(r'(?is)<(section|div)[^>]+(?:id|class)=["\'][^"\']*(contact|footer|reach|get-in-touch|اتصل|تواصل)[^"\']*["\'][^>]*>(.*?)</\1>', raw):
            contact_blocks.append(self._strip_tags(m.group(3)))
        contact = " ".join(contact_blocks[:2])

        # Prefer og:site_name as part of "title" signal (not replacing title)
        meta_combo = " ".join([x for x in [meta, og_site, og_title] if x])

        return {
            "title": title,
            "meta": meta_combo,
            "h1": h1,
            "h2": h2,
            "footer": footer,
            "contact": contact,
        }

    def _extract_jsonld(self, html: str) -> Dict[str, object]:
        """
        Extract first relevant Organization/LocalBusiness JSON-LD (best-effort).
        """
        raw = html
        scripts = re.findall(r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', raw)
        for s in scripts[:6]:
            try:
                txt = s.strip()
                if not txt:
                    continue
                data = json.loads(txt)
                # JSON-LD can be list or dict
                items = data if isinstance(data, list) else [data]
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    t = it.get("@type") or it.get("type")
                    # @type might be list
                    types = []
                    if isinstance(t, list):
                        types = [str(x) for x in t]
                    elif isinstance(t, str):
                        types = [t]
                    types_lower = [x.lower() for x in types]
                    if any(x in types_lower for x in ["organization", "localbusiness", "store", "professionalservice"]):
                        return it
            except Exception:
                continue
        return {}

    def _strip_tags(self, s: str) -> str:
        s = re.sub(r"(?s)<[^>]+>", " ", s)
        s = html_lib.unescape(s)
        s = s.replace("\xa0", " ")
        return re.sub(r"\s+", " ", s).strip()

    # -----------------------------
    # Name / Location extraction
    # -----------------------------

    def _extract_company_name(self, domain: str, norm_sections: Dict[str, str], jsonld: Dict[str, object]) -> str:
        # 1) JSON-LD
        name = ""
        if isinstance(jsonld, dict):
            n = jsonld.get("name")
            if isinstance(n, str) and n.strip():
                name = n.strip()

        if name:
            return self._clean_company_name(name)

        # 2) og:site_name / meta combo often includes site name
        meta = norm_sections.get("meta", "")
        # not great as normalized; better use original sections? We'll fallback to title/h1 anyway.

        # 3) title
        title = norm_sections.get("title", "")
        if title:
            # split by common separators; take leftmost chunk
            raw_title = title
            # title is normalized; still OK
            parts = re.split(r"\s*(?:\||-|–|—|:|::)\s*", raw_title)
            if parts and parts[0].strip():
                return self._clean_company_name(parts[0].strip())

        # 4) h1
        h1 = norm_sections.get("h1", "")
        if h1 and len(h1) >= 3:
            return self._clean_company_name(h1.strip())

        # 5) fallback: domain label
        base = domain.split(":")[0]
        base = base.split(".")[0] if base else domain
        return self._clean_company_name(base)

    def _clean_company_name(self, s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        # remove very generic suffix-only names
        s = re.sub(r"\b(home|homepage|official|website)\b", "", s, flags=re.IGNORECASE).strip()
        return s if s else "unknown"

    def _extract_city(self, norm_body: str, norm_sections: Dict[str, str], jsonld: Dict[str, object]) -> str:
        # 1) JSON-LD addressLocality
        if isinstance(jsonld, dict):
            addr = jsonld.get("address")
            if isinstance(addr, dict):
                loc = addr.get("addressLocality")
                if isinstance(loc, str) and loc.strip():
                    return self._normalize_city(loc.strip())

        # 2) Heuristic: count city mentions in strong sections first
        strong_text = " ".join([norm_sections.get("title", ""), norm_sections.get("h1", ""), norm_sections.get("contact", ""), norm_sections.get("footer", "")])
        best_city = ""
        best_score = 0

        for c in self.cities_ksa:
            cn = self._normalize(c)
            if not cn:
                continue
            # weighted: strong sections count *2 + body count
            strong_hits = strong_text.count(cn)
            body_hits = norm_body.count(cn)
            score = 2 * strong_hits + body_hits
            if score > best_score:
                best_score = score
                best_city = c

        if best_score > 0 and best_city:
            return self._normalize_city(best_city)

        return "unknown"

    def _normalize_city(self, s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        return s.lower() if s else "unknown"

    def _extract_country(self, norm_body: str, jsonld: Dict[str, object], domain: str) -> str:
        # JSON-LD country
        if isinstance(jsonld, dict):
            addr = jsonld.get("address")
            if isinstance(addr, dict):
                c = addr.get("addressCountry")
                if isinstance(c, str) and c.strip():
                    return c.strip()
                if isinstance(c, dict):
                    n = c.get("name")
                    if isinstance(n, str) and n.strip():
                        return n.strip()

        # heuristic
        if domain.endswith(".sa") or "saudi" in norm_body or "ksa" in norm_body or "السعود" in norm_body:
            return "Saudi Arabia"
        return "unknown"

    # -----------------------------
    # Scoring helpers
    # -----------------------------

    def _count_terms_weighted(self, text_map: Dict[str, str], terms: List[str], weights: Dict[str, float]) -> float:
        """
        Count occurrences of terms across multiple text sections with weights.
        Uses safe substring counts (fast) + tries word-boundary regex for latin terms.
        """
        if not terms:
            return 0.0

        total = 0.0
        for section, txt in text_map.items():
            if not txt:
                continue
            w = float(weights.get(section, 1.0))

            # For performance: split terms into "latinish single-word" vs others
            for t in terms:
                if not t:
                    continue
                tn = t.lower()
                # Use a cached compiled pattern for latin single-ish tokens to reduce false positives
                if re.fullmatch(r"[a-z0-9][a-z0-9 .+\-]{0,64}", tn) and any(ch.isalpha() for ch in tn):
                    # word-ish boundary for first/last; keep spaces allowed in phrase
                    pat = self._compile_term_pattern(tn)
                    hits = len(pat.findall(txt))
                else:
                    hits = txt.count(tn)

                if hits:
                    total += w * float(hits)

        return float(total)

    def _compile_term_pattern(self, term: str) -> re.Pattern:
        """
        Compile a somewhat-safe regex pattern for a term (latin phrases included).
        """
        key = term
        cache = self._compiled_terms_cache.get(key)
        if cache:
            return cache[0]

        escaped = re.escape(term)
        # If the term contains spaces, allow flexible whitespace
        escaped = escaped.replace(r"\ ", r"\s+")
        # Word boundaries at edges where reasonable
        pattern = rf"(?i)(?<![a-z0-9]){escaped}(?![a-z0-9])"
        pat = re.compile(pattern)
        self._compiled_terms_cache[key] = [pat]
        return pat

    def _log_norm(self, x: float, cap: float) -> float:
        if x <= 0:
            return 0.0
        return float(min(1.0, math.log1p(x) / math.log1p(cap)))

    def _word_count(self, norm_text: str) -> int:
        # EN+AR-ish tokens
        toks = re.findall(r"[a-zA-Z]{2,}|[\u0600-\u06FF]{2,}", norm_text)
        return int(len(toks))

    # -----------------------------
    # Legacy/compat helpers
    # -----------------------------

    def _bucket_signal(self, norm_text: str, terms: List[str], min_hits: int = 2) -> float:
        hits = 0
        for t in terms:
            if t and t.lower() in norm_text:
                hits += 1
                if hits >= min_hits:
                    return 1.0
        return 0.0

    def _ksa_signal(self, norm_text: str, contacts: Dict[str, List[str]], domain: str, city_hint: str = "unknown") -> float:
        # +966 is strong
        if any(p.startswith("+966") for p in contacts.get("phones", [])):
            return 1.0

        # country keywords
        if "saudi" in norm_text or "ksa" in norm_text or "السعود" in norm_text:
            return 1.0

        # city mention (prefer detected city)
        if city_hint and city_hint != "unknown":
            # if city itself is KSA city, this is strong enough
            if city_hint in [self._normalize_city(c) for c in self.cities_ksa]:
                return 0.85

        for c in self.cities_ksa:
            if self._normalize(c) in norm_text:
                return 0.8

        # domain hint
        if domain.endswith(".sa"):
            return 0.7

        return 0.0

    def _non_ksa_signal(self, norm_text: str, ksa_signal: float, domain: str = "") -> bool:
        # If we already have decent KSA evidence, don't penalize
        if ksa_signal >= 0.7:
            return False

        if domain.endswith(".ae"):
            return True

        non_ksa_terms = [
            # GCC / neighbors
            "uae", "dubai", "abu dhabi", "qatar", "doha", "kuwait",
            "oman", "bahrain", "egypt", "cairo", "jordan", "amman",
            "lebanon", "beirut", "iraq", "baghdad", "syria", "damascus",
            "yemen", "sana'a", "sanaa", "morocco", "rabat", "tunisia",
            "algeria", "libya", "sudan",
            # Wider non-KSA signals
            "turkey", "istanbul", "uk", "london", "usa", "new york", "germany", "berlin",
            # Arabic
            "الإمارات", "دبي", "ابوظبي", "أبوظبي",
            "قطر", "الدوحة", "الكويت", "عمان", "البحرين",
            "مصر", "القاهرة", "الأردن", "عمّان",
            "لبنان", "بيروت", "العراق", "بغداد",
            "سوريا", "دمشق", "اليمن", "صنعاء",
            "المغرب", "الرباط", "تونس", "الجزائر", "ليبيا", "السودان",
        ]
        for t in non_ksa_terms:
            if t in norm_text:
                return True
        return False

    # -----------------------------
    # Utils
    # -----------------------------

    def _normalize(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _domain(self, url: str) -> str:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host

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
    # Keyword defaults
    # -----------------------------

    def _default_keywords(self) -> Dict[str, List[str]]:
        """
        Keep your original buckets (kitchen/interior/fitout/portfolio/showroom/designer/architect/developer/individual_markers/business_markers)
        and add new buckets:
          - high_precision
          - mid_precision
          - business_intent
        """
        base = {
            # Original (kept, unchanged as much as possible)
            "kitchen": [
                "kitchen", "kitchens", "kitchen design", "kitchen installation",
                "kitchen supplier", "kitchen company", "kitchen showroom",
                "cabinet", "cabinets", "cabinetry", "kitchen cabinets",
                "wardrobe", "wardrobes", "closet", "closets",
                "fitted wardrobe", "built-in wardrobe", "walk-in closet",
                "joinery", "woodwork", "carpentry", "millwork",
                "modular kitchen", "custom kitchen", "custom cabinetry",
                "countertop", "worktop", "granite", "quartz", "marble",
                "vanity", "vanities", "tv unit", "wall unit",
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
                "interior", "interiors", "interior design", "interior designer",
                "design studio", "design consultancy", "space planning",
                "decoration", "decor", "home decor", "styling",
                "furniture design", "custom furniture", "bespoke furniture",
                "residential interior", "commercial interior",
                "تصميم داخلي", "مصمم داخلي", "استشارات تصميم", "استشارات ديكور",
                "استديو تصميم", "تخطيط المساحات", "توزيع المساحات",
                "ديكور", "ديكورات", "ديكور منزلي", "تنسيق",
                "تصميم أثاث", "أثاث تفصيل", "تفصيل أثاث", "أثاث حسب الطلب",
                "تصميم سكني", "تصميم تجاري",
            ],
            "fitout": [
                "fit-out", "fit out", "fitout", "interior fit-out",
                "turnkey", "turn key", "turn-key", "turnkey solutions",
                "interior finishing", "construction finishing",
                "villa finishing", "office fit-out", "commercial fit-out",
                "renovation", "remodel", "refurbishment",
                "contracting", "contractor", "general contractor",
                "civil works", "gypsum", "false ceiling", "partition",
                "mep", "electrical", "plumbing", "hvac",
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
                "portfolio", "projects", "case study", "case studies",
                "gallery", "project gallery", "before and after",
                "completed projects", "recent projects",
                "مشاريع", "مشروع", "اعمالنا", "أعمالنا", "نماذج أعمال",
                "معرض اعمال", "معرض أعمال", "ألبوم", "قبل وبعد",
                "مشاريع منفذة", "أحدث المشاريع",
            ],
            "showroom": [
                "showroom", "kitchen showroom", "display", "exhibition",
                "experience center", "store", "shop",
                "معرض", "صالة عرض", "مركز تجربة", "متجر", "محل",
                "عرض", "معرض مطابخ",
            ],
            "designer": [
                "interior designer", "interior design", "design consultancy",
                "space planning", "concept design", "moodboard",
                "مصمم داخلي", "تصميم داخلي", "استشارات تصميم",
                "تخطيط المساحات", "تصميم مفاهيمي", "لوحة إلهام",
            ],
            "architect": [
                "architect", "architecture", "architectural", "architects",
                "engineering office", "engineering consultancy",
                "architectural design", "structural design",
                "site supervision", "permit", "building permit",
                "معماري", "هندسة معمارية", "تصميم معماري", "مكتب هندسي",
                "استشارات هندسية", "تصميم إنشائي", "هندسة إنشائية",
                "إشراف", "إشراف هندسي", "رخصة بناء", "تصريح بناء",
                "مخططات", "مخططات معمارية", "اعتماد مخططات",
            ],
            "developer": [
                "real estate developer", "property developer", "development company",
                "real estate development", "projects development",
                "مطوّر عقاري", "مطور عقاري", "تطوير عقاري", "شركة تطوير",
                "تطوير مشاريع", "تطوير عقاري سكني", "تطوير عقاري تجاري",
            ],
            "individual_markers": [
                "i need", "looking for", "need a", "want to", "can someone",
                "my villa", "my apartment", "my house", "my kitchen",
                "budget", "price", "how much", "cost",
                "whatsapp me", "call me", "dm me", "contact me",
                "احتاج", "أحتاج", "ابحث عن", "أبحث عن", "محتاج", "اريد", "أريد",
                "بيتي", "شقتي", "فيلا", "مطبخي", "مطبخ",
                "ميزانية", "سعر", "اسعار", "تكلفة", "كم السعر", "كم يكلف",
                "واتساب", "راسلني", "اتصل علي", "كلموني", "رسالة خاصة",
            ],
            "business_markers": [
                "llc", "ltd", "company", "co.", "corp", "inc",
                "contracting", "contractor", "trading", "factory", "manufacturer",
                "since", "est.", "established", "branches", "head office",
                "commercial", "b2b", "profile", "company profile",
                "iso", "vat", "cr number", "registration",
                "شركة", "مؤسسة", "مجموعة", "مصنع", "شركة تصنيع",
                "مقاول", "مقاولات", "شركة مقاولات", "مقاول عام",
                "تجارة", "تجارية", "توريد",
                "تأسست", "تأسيس", "منذ", "فرع", "فروع", "المقر الرئيسي",
                "سجل تجاري", "رقم السجل", "رقم ضريبي", "الرقم الضريبي",
                "بروفايل", "ملف الشركة", "شهادة", "ايزو",
                "مكتب هندسي", "استشارات هندسية",
            ],
        }

        # New: higher-precision terms (more discriminative)
        base["high_precision"] = [
            # EN
            "kitchen showroom", "kitchen installation", "modular kitchen", "custom cabinetry", "kitchen cabinets",
            "interior fit-out", "fit out contractor", "turnkey fit-out", "turnkey solutions",
            "villa finishing", "office fit-out", "commercial fit-out",
            "joinery", "bespoke furniture",
            # AR
            "معرض مطابخ", "تفصيل مطابخ", "تركيب مطابخ",
            "فيت اوت", "تشطيب داخلي", "تسليم مفتاح",
            "شركة مقاولات", "مقاول تشطيبات",
        ]

        # New: mid-precision terms (supporting evidence)
        base["mid_precision"] = [
            # EN
            "kitchen", "cabinet", "cabinetry", "wardrobe", "closet", "walk-in closet",
            "interior", "interior design", "renovation", "refurbishment", "contractor",
            "showroom", "projects", "portfolio", "fitout", "finishing",
            # AR
            "مطابخ", "خزائن", "دواليب", "غرف ملابس",
            "تصميم داخلي", "ديكور", "ترميم", "تجديد", "مقاولات", "تشطيب",
            "مشاريع", "أعمالنا", "معرض",
        ]

        # New: business-intent / proof-of-service
        base["business_intent"] = [
            # EN
            "services", "our services", "service", "request a quote", "get a quote", "contact us",
            "our projects", "completed projects", "portfolio", "case studies", "company profile",
            "branches", "head office", "factory", "manufacturer",
            # AR
            "خدمات", "اطلب عرض سعر", "عرض سعر", "اتصل بنا", "تواصل معنا",
            "أعمالنا", "مشاريع", "ملف الشركة", "بروفايل", "فروع", "المقر الرئيسي",
        ]

        return base

    def _default_negative_keywords(self) -> List[str]:
        return [
            # EN (irrelevant areas)
            "car repair", "automotive", "garage", "tyres", "tire", "car dealership", "used cars",
            "restaurant", "cafe", "bakery", "catering",
            "hospital", "clinic", "dentist", "pharmacy",
            "hotel", "booking", "travel agency", "tourism",
            "food delivery", "delivery app", "restaurant delivery",
            "real estate listing", "property listing", "classifieds", "marketplace",
            "crypto", "forex", "bitcoin", "trading platform",
            "used cars", "car dealer",
            "hair salon", "barber", "spa",
            # common strong non-target verticals (practical)
            "bank", "banking", "loan", "credit card", "insurance",
            "pest control", "exterminator", "termite",
            "cement", "concrete", "steel", "rebar", "asphalt",

            # AR (irrelevant areas)
            "ورشة سيارات", "صيانة سيارات", "ميكانيكا", "كهرباء سيارات", "إطارات", "معرض سيارات",
            "مطعم", "كافيه", "مقهى", "مخبز", "تموين",
            "مستشفى", "عيادة", "طبيب أسنان", "صيدلية",
            "فندق", "حجز", "سفر", "وكالة سفر", "سياحة",
            "توصيل الطعام", "تطبيق توصيل", "طلبات طعام",
            "عقارات", "إعلانات عقارية", "منصة إعلانات",
            "عملات رقمية", "تداول", "فوركس", "بيتكوين", "منصة تداول",
            "سيارات مستعملة", "معرض سيارات",
            "صالون", "حلاق", "سبا",
            # strong non-target verticals (AR)
            "بنك", "قرض", "بطاقة", "تأمين",
            "مكافحة حشرات", "رش", "نمل", "صراصير",
            "اسمنت", "خرسانة", "حديد", "قضبان",
        ]


# -----------------------------
# Example usage (manual testing)
# -----------------------------
if __name__ == "__main__":
    urls = [
        "https://www.nadco.com.sa/en/",
        "https://www.perfectartcontracting.com/",
        "https://spacearabia.sa/premium-fitout-saudi-arabia/",
        "https://sunshinepestcontrol.sa/kitchen-cupboard-cleaning-tips/",
        "https://www.riyadbank.com/",
    ]
    for url in urls:
        ev = SiteEvaluator().evaluate_url(url=url)
        print(ev)
        for reason in ev.reasons:
            print("-", reason)
