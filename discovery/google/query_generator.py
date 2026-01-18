from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class QuerySpec:
    q: str
    tag: str  # for grouping (kitchen/fitout/architect/etc)
    city: str


class QueryGenerator:
    """
    Generates query strings for Google search.
    """
    DEFAULT_CITIES = [
        "Riyadh", "Jeddah", "Dammam", "Khobar", "Makkah", "Madinah",
    ]

    KEYWORDS = {
        "kitchen_en": [
            "kitchen cabinets",
            "kitchen showroom",
            "modular kitchen",
            "custom kitchen",
            "kitchen installation",
            "wardrobes",
            "walk-in closet",
        ],
        "kitchen_ar": [
            "مطابخ",
            "تفصيل مطابخ",
            "تركيب مطابخ",
            "خزائن مطبخ",
            "غرف ملابس",
            "دواليب",
        ],
        "fitout_en": [
            "fit out contractor",
            "interior fit out",
            "turnkey interior",
            "interior finishing contractor",
        ],
        "fitout_ar": [
            "تشطيب",
            "مقاول تشطيبات",
            "تسليم مفتاح",
            "فيت اوت",
        ],
        "architect_en": [
            "engineering consultancy",
            "architectural design office",
            "engineering office",
        ],
        "architect_ar": [
            "مكتب هندسي",
            "استشارات هندسية",
            "تصميم معماري",
        ],
        # .. (other lead types)
    }

    def __init__(self, cities: Sequence[str] | None = None):
        self.cities = list(cities) if cities else list(self.DEFAULT_CITIES)

    def generate(self, include_tags: Sequence[str] | None = None) -> List[QuerySpec]:
        tags = list(include_tags) if include_tags else list(self.KEYWORDS.keys())
        out: List[QuerySpec] = []

        for city in self.cities:
            for tag in tags:
                for kw in self.KEYWORDS[tag]:
                    out.append(QuerySpec(q=f"{kw} {city}", tag=tag, city=city))

        return out

    @staticmethod
    def export_to_txt(queries: Iterable[QuerySpec], path: str = "google_queries.txt") -> None:
        lines = []
        for qs in queries:
            lines.append(qs.q)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
