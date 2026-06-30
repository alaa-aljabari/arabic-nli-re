"""
templates.py
------------
Relation-aware verbalization templates for the NLI-RE framework.

Section 3 (§Template Construction) defines a template T_r(e1, e2) for each
relation type r : D → Z that verbalizes a candidate relation between subject
entity e1 and object entity e2 as an Arabic hypothesis string h.

Each template is a callable: (subject, object) -> hypothesis string.
"""

from typing import Callable, Dict, FrozenSet, Tuple

# Type alias
RelationTemplate = Callable[[str, str], str]

RELATION_TEMPLATES: Dict[str, RelationTemplate] = {
    # =============== Family =========================================================
    # {e1} هو والد أو والدة {e2}       → is the parent of
    "Family.has_parent":               lambda e1, e2: f"{e1} هو والد أو والدة {e2}",
    # {e1} هو أخ أو أخت {e2}           → is the sibling of
    "Family.has_sibling":              lambda e1, e2: f"{e1} هو أخ أو أخت {e2}",
    # {e1} هو زوج أو زوجة {e2}         → is the spouse of
    "Family.has_spouse":               lambda e1, e2: f"{e1} هو زوج أو زوجة {e2}",
    # {e1} هو قريب {e2}                → is a relative of
    "Family.has_relative":             lambda e1, e2: f"{e1} هو قريب {e2}",

    # =============== Personal =========================================================
    # {e1} وُلِد في تاريخ {e2}          → was born on
    "Personal.birth_date":             lambda e1, e2: f"{e1} وُلِد في تاريخ {e2}",
    # {e1} توفي في تاريخ {e2}           → died on
    "Personal.death_date":             lambda e1, e2: f"{e1} توفي في تاريخ {e2}",
    # {e1} وُلِد في / مكان الولادة {e2} → was born in
    "Personal.birth_place":            lambda e1, e2: f"{e1} وُلِد في / مكان الولادة {e2}",
    # {e1} يعمل كـ / مهنته {e2}        → works as
    "Personal.has_occupation":         lambda e1, e2: f"{e1} يعمل كـ / مهنته {e2}",

    # =============== Business =========================================================
    # {e1} لديه نزاع مع {e2}           → has a conflict with
    "Business.has_conflict_with":      lambda e1, e2: f"{e1} لديه نزاع مع {e2}",
    # {e1} منافس لـ {e2}               → is a competitor of
    "Business.has_competitor":         lambda e1, e2: f"{e1} منافس لـ {e2}",
    # {e1} شريك لـ {e2}                → is a partner of
    "Business.has_partner_with":       lambda e1, e2: f"{e1} شريك لـ {e2}",

    # =============== Administration =========================================================
    # {e1} هو مدير {e2}                → is the manager of
    "Administration.manager_of":       lambda e1, e2: f"{e1} هو مدير {e2}",
    # {e1} هو رئيس/يتولى أعلى منصب في {e2} → is the president of
    "Administration.president_of":     lambda e1, e2: f"{e1} هو رئيس/يتولى أعلى منصب في {e2}",
    # {e1} هو قائد {e2}                → is the leader of
    "Administration.leader_of":        lambda e1, e2: f"{e1} هو قائد {e2}",

    # =============== PartOf =========================================================
    # {e1} هو تقسيم جغرافي لـ {e2}     → is a geopolitical division of
    "PartOf.geopolitical_division":    lambda e1, e2: f"{e1} هو تقسيم جغرافي لـ {e2}",
    # {e1} فرع تابع لـ {e2}            → is a subsidiary of
    "PartOf.subsidiary":               lambda e1, e2: f"{e1} فرع تابع لـ {e2}",

    # =============== Affiliation =========================================================
    # {e1} عضو في {e2}                 → is a member of
    "Affiliation.member_of":           lambda e1, e2: f"{e1} عضو في {e2}",
    # {e1} يعمل لدى {e2}               → is employed by
    "Affiliation.employee_of":         lambda e1, e2: f"{e1} يعمل لدى {e2}",
    # {e1} طالب في / تلقى تعليمه في {e2} → studies at
    "Affiliation.student_at":          lambda e1, e2: f"{e1} طالب في / تلقى تعليمه في {e2}",
    # {e1} يمتلك {e2}                  → owns
    "Affiliation.owner_of":            lambda e1, e2: f"{e1} يمتلك {e2}",

    # =============== Productivity =========================================================
    # {e1} مخترع {e2}                  → is the inventor of
    "Productivity.inventor_of":        lambda e1, e2: f"{e1} مخترع {e2}",
    # {e1} يصنّع {e2}                  → manufactures
    "Productivity.manufacturer_of":    lambda e1, e2: f"{e1} يصنّع {e2}",
    # {e1} بنى {e2}                    → built
    "Productivity.builder_of":         lambda e1, e2: f"{e1} بنى {e2}",
    # {e1} هو مؤسس {e2}               → is the founder of
    "Productivity.founder_of":         lambda e1, e2: f"{e1} هو مؤسس {e2}",

    # =============== Location =========================================================
    # {e1} يعيش في {e2}               → lives in
    "Location.lives_in":               lambda e1, e2: f"{e1} يعيش في {e2}",
    # {e1} يقع في {e2}                → is located in
    "Location.located_in":             lambda e1, e2: f"{e1} يقع في {e2}",
    # {e1} يقع مقره الرئيسي في {e2}   → is headquartered in
    "Location.headquartered_in":       lambda e1, e2: f"{e1} يقع مقره الرئيسي في {e2}",
    # {e1} لديه حدود مع {e2}          → borders
    "Location.has_border_with":        lambda e1, e2: f"{e1} لديه حدود مع {e2}",
    # {e1} يقع بالقرب من {e2}         → is near
    "Location.nearby":                 lambda e1, e2: f"{e1} يقع بالقرب من {e2}",

    # =============== Organization =========================================================
    # {e1} لديه ممتلكات {e2}           → has property
    "Organization.has_propoerty":      lambda e1, e2: f"{e1} لديه ممتلكات {e2}",
    # {e1} يضم عدد فروع قدره {e2}      → has branches
    "Organization.branch_count":       lambda e1, e2: f"{e1} يضم عدد فروع قدره {e2}",
    # {e1} يحقق إيرادات قدرها {e2}    → generates revenue of
    "Organization.has_revenue":        lambda e1, e2: f"{e1} يحقق إيرادات قدرها {e2}",
    # {e1} عدد موظفيه {e2}             → employs employees
    "Organization.employs":            lambda e1, e2: f"{e1} عدد موظفيه {e2}",
    # {e1} تم تأسيسها بتاريخ {e2}     → was founded on
    "Organization.found_on":           lambda e1, e2: f"{e1} تم تأسيسها بتاريخ {e2}",
    # {e1} يُعرف أيضاً باسم {e2}      → is also known as
    "Organization.has_alternate_name": lambda e1, e2: f"{e1} يُعرف أيضاً باسم {e2}",

    # =============== GPE (Geo-Political Entity) =========================================================
    # {e1} تبلغ مساحتها {e2}           → has an area of
    "GPE.has_area":                    lambda e1, e2: f"{e1} تبلغ مساحتها {e2}",
    # {e1} لغتها الرسمية {e2}          → official language is
    "GPE.official_language":           lambda e1, e2: f"{e1} لغتها الرسمية {e2}",
    # {e1} عملتها هي {e2}              → has currency
    "GPE.has_currency":                lambda e1, e2: f"{e1} عملتها هي {e2}",
    # {e1} هي عاصمة {e2}              → is the capital of
    "GPE.capital_of":                  lambda e1, e2: f"{e1} هي عاصمة {e2}",
    # {e1} عدد سكانها {e2}             → has a population of
    "GPE.has_population":              lambda e1, e2: f"{e1} عدد سكانها {e2}",
}

# Relation schema
RELATION_SCHEMA: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    # Family
    "Family.has_parent":               (frozenset({"PERS"}),               frozenset({"PERS"})),
    "Family.has_spouse":               (frozenset({"PERS"}),               frozenset({"PERS"})),
    "Family.has_sibling":              (frozenset({"PERS"}),               frozenset({"PERS"})),
    "Family.has_relative":             (frozenset({"PERS"}),               frozenset({"PERS"})),
    # Personal
    "Personal.birth_date":             (frozenset({"PERS"}),               frozenset({"DATE"})),
    "Personal.death_date":             (frozenset({"PERS"}),               frozenset({"DATE"})),
    "Personal.birth_place":            (frozenset({"PERS"}),               frozenset({"GPE", "LOC"})),
    "Personal.has_occupation":         (frozenset({"PERS"}),               frozenset({"OCC"})),
    # Business
    "Business.has_conflict_with":      (frozenset({"ORG", "NORP", "GPE"}), frozenset({"ORG", "NORP", "GPE"})),
    "Business.has_competitor":         (frozenset({"PERS", "ORG"}),        frozenset({"PERS", "ORG"})),
    "Business.has_partner_with":       (frozenset({"ORG"}),                frozenset({"ORG"})),
    # Administration
    "Administration.manager_of":       (frozenset({"PERS"}),               frozenset({"ORG", "FAC"})),
    "Administration.president_of":     (frozenset({"PERS"}),               frozenset({"ORG", "GPE"})),
    "Administration.leader_of":        (frozenset({"PERS"}),               frozenset({"ORG"})),
    # PartOf
    "PartOf.geopolitical_division":    (frozenset({"GPE", "LOC"}),         frozenset({"GPE", "LOC"})),
    "PartOf.subsidiary":               (frozenset({"ORG"}),                frozenset({"ORG"})),
    # Affiliation
    "Affiliation.member_of":           (frozenset({"PERS", "GPE"}),        frozenset({"ORG", "NORP"})),
    "Affiliation.employee_of":         (frozenset({"PERS"}),               frozenset({"ORG", "FAC"})),
    "Affiliation.student_at":          (frozenset({"PERS"}),               frozenset({"ORG"})),
    "Affiliation.owner_of":            (frozenset({"PERS"}),               frozenset({"ORG", "FAC"})),
    # Productivity
    "Productivity.inventor_of":        (frozenset({"PERS"}),               frozenset({"PRODUCT"})),
    "Productivity.manufacturer_of":    (frozenset({"ORG"}),                frozenset({"PRODUCT"})),
    "Productivity.builder_of":         (frozenset({"PERS", "NORP", "ORG"}),frozenset({"FAC", "ORG"})),
    "Productivity.founder_of":         (frozenset({"PERS"}),               frozenset({"ORG"})),
    # Location
    "Location.lives_in":               (frozenset({"PERS", "NORP"}),       frozenset({"GPE", "LOC"})),
    "Location.located_in":             (frozenset({"FAC", "ORG"}),         frozenset({"GPE", "LOC"})),
    "Location.headquartered_in":       (frozenset({"ORG"}),                frozenset({"LOC", "GPE"})),
    "Location.has_border_with":        (frozenset({"LOC", "GPE"}),         frozenset({"LOC", "GPE"})),
    "Location.nearby":                 (frozenset({"GPE", "LOC", "FAC"}),  frozenset({"GPE", "LOC", "FAC"})),
    # Organization
    "Organization.has_propoerty":      (frozenset({"ORG"}),                frozenset({"PRODUCT"})),
    "Organization.branch_count":       (frozenset({"ORG"}),                frozenset({"CARDINALITY"})),
    "Organization.has_revenue":        (frozenset({"ORG"}),                frozenset({"MONEY"})),
    "Organization.employs":            (frozenset({"ORG"}),                frozenset({"CARDINALITY"})),
    "Organization.found_on":           (frozenset({"ORG"}),                frozenset({"DATE", "TIME"})),
    "Organization.has_alternate_name": (frozenset({"ORG", "FAC"}),         frozenset({"ORG", "FAC"})),
    # GPE
    "GPE.has_area":                    (frozenset({"GPE", "LOC"}),         frozenset({"QUANTITY"})),
    "GPE.official_language":           (frozenset({"GPE", "LOC"}),         frozenset({"LANGUAGE"})),
    "GPE.has_currency":                (frozenset({"GPE", "LOC"}),         frozenset({"CURRENCY"})),
    "GPE.has_population":              (frozenset({"GPE"}),                frozenset({"CARDINALITY"})),
    "GPE.capital_of":                  (frozenset({"GPE"}),                frozenset({"GPE"})),
}

# All relation types that carry a positive entailment signal
POSITIVE_RELATIONS = set(RELATION_TEMPLATES.keys())


def verbalize(relation: str, subject: str, obj: str) -> str:
    """
    Verbalize a candidate relation between subject and object into an
    Arabic hypothesis string h using the predefined template T_r(e1, e2).

    Args:
        relation: Relation type string (must be a key in RELATION_TEMPLATES).
        subject:  Subject entity mention n_i.
        obj:      Object entity mention n_j.

    Returns:
        Arabic hypothesis string h.

    Raises:
        KeyError: If relation is not in RELATION_TEMPLATES.
    """
    if relation not in RELATION_TEMPLATES:
        raise KeyError(
            f"No template defined for relation '{relation}'. "
            f"Available: {sorted(RELATION_TEMPLATES.keys())}"
        )
    return RELATION_TEMPLATES[relation](subject, obj)


def build_nli_sentence(premise: str, hypothesis: str) -> str:
    """
    Construct the full NLI input string as defined in Eq. 1:

        H = T([CLS] s [SEP] h)

    Args:
        premise:    Input sentence s containing the entity pair.
        hypothesis: Verbalized relation hypothesis h = T_r(e1, e2).

    Returns:
        Concatenated string: "[CLS] {premise} [SEP] {hypothesis}"
    """
    return f"[CLS] {premise} [SEP] {hypothesis}"
