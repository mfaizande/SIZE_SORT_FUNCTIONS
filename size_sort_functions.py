"""
size_sort_functions.py
======================
One named sort function per SIZE_GROUPING.

Formula for every group:
    gt_size_order = rank * 10 + IDENTIFIER_NUMBERS[size_grouping]

Usage
-----
from size_sort_functions import build_size_order_df, SIZE_GROUPING_SORT_MAP

df = build_size_order_df(size_grouping='kids_age', sizes=['3 Monate', '1 Jahr', 'Frühchen'])

SORTING_METHOD column for Variant_meta_data
-------------------------------------------
SIZE_GROUPING_SORT_MAP maps each size_grouping name to its function name string,
which you can write directly into the SORTING_METHOD column.
"""

import re
import pandas as pd
import numpy as np
from fractions import Fraction
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IDENTIFIER_NUMBERS = {
    'apparel_number':    1_000_000,
    'apparel_standard':  1_100_000,
    'apparel_num-range': 1_200_000,
    'unterwäsche':       1_300_000,
    'waist-length':      1_400_000,
    'shoes_number':      1_500_000,
    'shoes_variant':     1_600_000,
    'kids_age':          1_700_000,
    'watch_size':        1_800_000,
    'jewellery_size':    1_900_000,
    'headwear_size':     2_000_000,
    'belt_size':         2_100_000,
    'baggage_size':      2_200_000,
    'eyewear_size':      2_300_000,
    'one_size':          2_400_000,
    'shoes_number_uk':   2_500_000,
}

# 'Einheitsgröße' always gets this absolute order — injected into every group
EINHEITSGROESSE_GT_SIZE  = 'Einheitsgröße'
EINHEITSGROESSE_GT_ORDER = 1_000_000_000
EINHEITSGROESSE_GROUPING = 'one_size'


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_df(sorted_sizes: list, group_name: str) -> pd.DataFrame:
    """
    Assign gt_size_order to an already-sorted list and append Einheitsgröße.
    gt_size_order = (rank * 10) + IDENTIFIER_NUMBERS[group_name]
    """
    identifier = IDENTIFIER_NUMBERS[group_name]
    rows = [
        {'gt_size': s, 'gt_size_order': (i + 1) * 10 + identifier, 'size_grouping': group_name}
        for i, s in enumerate(sorted_sizes)
    ]
    rows.append({
        'gt_size': EINHEITSGROESSE_GT_SIZE,
        'gt_size_order': EINHEITSGROESSE_GT_ORDER,
        'size_grouping': EINHEITSGROESSE_GROUPING,
    })
    return pd.DataFrame(rows)


def _clean_deduplicate(sizes: list, normalize_fn=None) -> list:
    """
    Strip, optionally normalize, drop Einheitsgröße (injected separately),
    case-insensitive deduplication (keep first seen), report duplicates.
    """
    sizes = [str(s).strip() for s in sizes if pd.notna(s)]
    if normalize_fn:
        sizes = [normalize_fn(s) for s in sizes]

    sizes = [s for s in sizes if s.lower() != 'einheitsgröße']

    counts = Counter(s.lower() for s in sizes)
    dupes  = {s: c for s, c in counts.items() if c > 1}
    if dupes:
        print(f'  Duplicates removed:')
        for s, c in sorted(dupes.items()):
            print(f'    "{s}" × {c}')

    seen, unique = {}, []
    for s in sizes:
        key = s.lower()
        if key not in seen:
            seen[key] = True
            unique.append(s)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# 1. apparel_standard  — fixed dictionary lookup
# ─────────────────────────────────────────────────────────────────────────────

_APPAREL_STANDARD_RANK = {
    'XXS': 1, 'XXS-XS': 2, 'XS': 3, 'XS-S': 4, 'XS-XL': 5,
    'S': 6, 'S-M': 7, 'S-L': 8,
    'M': 9, 'M-L': 10, 'M-XXL': 11,
    'L': 12, 'L-XL': 13, 'L-XXL': 14,
    'XL': 15, 'XL-XXL': 16,
    'XXL': 17, 'XXL-3XL': 18,
}

def apparel_standard_sort(sizes=None) -> pd.DataFrame:
    """
    Letter-based apparel sizes (XS, S, M, L, XL …).
    Sorted by a fixed rank dictionary.
    If sizes=None the full standard set is used.
    """
    if sizes is None:
        ordered = sorted(_APPAREL_STANDARD_RANK.keys(),
                         key=lambda s: _APPAREL_STANDARD_RANK[s])
    else:
        unique = _clean_deduplicate(sizes)
        ordered = sorted(unique,
                         key=lambda s: _APPAREL_STANDARD_RANK.get(s, 999))
    return _build_df(ordered, 'apparel_standard')


# ─────────────────────────────────────────────────────────────────────────────
# 2. apparel_number  — numeric ascending
# ─────────────────────────────────────────────────────────────────────────────

def apparel_number_sort(sizes: list) -> pd.DataFrame:
    """
    Numeric apparel sizes (34, 36, 38 …).
    Sorted as integers ascending.
    """
    unique  = _clean_deduplicate(sizes)
    ordered = sorted(unique, key=lambda s: int(s))
    return _build_df(ordered, 'apparel_number')


# ─────────────────────────────────────────────────────────────────────────────
# 3. apparel_num-range  — sort by (low, high) ascending
# ─────────────────────────────────────────────────────────────────────────────

def _num_range_key(s):
    parts = s.split('-')
    return (int(parts[0]), int(parts[1]))

def apparel_num_range_sort(sizes: list) -> pd.DataFrame:
    """
    Numeric range apparel sizes like '30-32', '44-46'.
    Sorted by (low ASC, high ASC).
    """
    unique  = _clean_deduplicate(sizes)
    ordered = sorted(unique, key=_num_range_key)
    return _build_df(ordered, 'apparel_num-range')


# ─────────────────────────────────────────────────────────────────────────────
# 4. unterwäsche  — (number ASC, letter ASC)
# ─────────────────────────────────────────────────────────────────────────────

def _unterwasche_key(s):
    m = re.match(r'(\d+)([A-Za-z]+)$', s)
    if m:
        return (int(m.group(1)), m.group(2))
    return (999_999, s)

def unterwasche_sort(sizes: list) -> pd.DataFrame:
    """
    Underwear/lingerie sizes like '80B', '90C', '100D'.
    Parsed as (number, cup_letter) and sorted (number ASC, letter ASC).
    """
    unique  = _clean_deduplicate(sizes)
    ordered = sorted(unique, key=_unterwasche_key)
    return _build_df(ordered, 'unterwäsche')


# ─────────────────────────────────────────────────────────────────────────────
# 5. waist-length  — (waist ASC, length ASC)  format: "{W}W/{L}L"
# ─────────────────────────────────────────────────────────────────────────────

def _waist_length_key(s):
    parts = s.split('/')
    waist  = int(parts[0].rstrip('W'))
    length = int(parts[1].rstrip('L'))
    return (waist, length)

def waist_length_sort(sizes: list) -> pd.DataFrame:
    """
    Jeans sizes like '30W/32L'.
    Sorted by (waist ASC, length ASC).
    """
    unique  = _clean_deduplicate(sizes)
    ordered = sorted(unique, key=_waist_length_key)
    return _build_df(ordered, 'waist-length')


# ─────────────────────────────────────────────────────────────────────────────
# 6. shoes_number  — numeric + width sub-category priority
# ─────────────────────────────────────────────────────────────────────────────

_SHOES_NUMBER_PRIORITY = {'Schmal': 0, 'number': 1, 'fraction': 1, 'Weit': 2, 'X-Weit': 3}

def _detect_shoes_subcat(s):
    raw = str(s).strip().lower()
    if 'x-weit' in raw or 'xweit' in raw:
        return 'X-Weit'
    if 'weit' in raw:
        return 'Weit'
    if 'schmal' in raw:
        return 'Schmal'
    if '/' in raw:
        return 'fraction'
    return 'number'

def _mixed_fraction_to_float(value):
    parts = str(value).split()
    if len(parts) == 2:
        return float(parts[0]) + float(Fraction(parts[1]))
    return float(Fraction(value))

def _shoes_number_key(s):
    subcat   = _detect_shoes_subcat(s)
    priority = _SHOES_NUMBER_PRIORITY[subcat]
    raw      = str(s).strip()
    numeric_part = raw.split()[0] if ' ' in raw else raw
    # strip any alpha suffix for width variants (e.g. "42 Weit" → "42")
    numeric_part = re.split(r'[A-Za-z]', numeric_part)[0]
    try:
        if subcat == 'fraction':
            num = _mixed_fraction_to_float(raw)
        else:
            num = float(numeric_part)
    except (ValueError, ZeroDivisionError):
        num = 999_999.0
    return (num, priority)

def shoes_number_sort(sizes: list) -> pd.DataFrame:
    """
    Shoe sizes with numbers, fractions ('38 1/2'), and width variants
    (Schmal, Weit, X-Weit).
    Sorted by (numeric_value ASC, width_priority ASC).
    """
    unique  = _clean_deduplicate(sizes)
    ordered = sorted(unique, key=_shoes_number_key)
    return _build_df(ordered, 'shoes_number')


# ─────────────────────────────────────────────────────────────────────────────
# 7. shoes_variant  — (val1 ASC, val2 ASC)  format: "38/39"
# ─────────────────────────────────────────────────────────────────────────────

def _shoes_variant_key(s):
    parts = s.split('/')
    return (float(parts[0]), float(parts[1]))

def shoes_variant_sort(sizes: list) -> pd.DataFrame:
    """
    Dual-value shoe sizes like '38/39', '40.5/41'.
    Sorted by (val1 ASC, val2 ASC).
    Format preserved with :g (drops trailing zeros).
    """
    unique  = _clean_deduplicate(sizes)
    # Normalize format: re-format after parsing to ensure consistent "38/39" not "38.0/39.0"
    def normalize(s):
        a, b = s.split('/')
        return f'{float(a):g}/{float(b):g}'
    unique  = [normalize(s) for s in unique]
    ordered = sorted(unique, key=_shoes_variant_key)
    return _build_df(ordered, 'shoes_variant')


# ─────────────────────────────────────────────────────────────────────────────
# 8. kids_age  — convert to months, sort (start_months, end_months, tier)
# ─────────────────────────────────────────────────────────────────────────────

_KIDS_SPECIAL = {'Frühchen': (-2, -2, 0)}

def _normalize_kids_size(s):
    s = str(s).strip()
    s = re.sub(r'(\d)(Jahr)',  r'\1 \2', s)
    s = re.sub(r'(\d)(Monat)', r'\1 \2', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def _kids_age_key(size):
    s = _normalize_kids_size(size)
    if s in _KIDS_SPECIAL:
        return _KIDS_SPECIAL[s]

    is_schmal = 3 if 'schmal' in s.lower() else (2 if 'jahr' in s.lower() else 1)

    if re.search(r'monat', s, re.I):
        unit = 1
    elif re.search(r'jahr', s, re.I):
        unit = 12
    else:
        return (999_999, 999_999, is_schmal)

    nums = re.findall(r'\d+', s)
    if not nums:
        return (999_999, 999_999, is_schmal)
    a = int(nums[0])
    b = int(nums[1]) if len(nums) > 1 else 0
    return (a * unit, b * unit, is_schmal)

def kids_age_sort(sizes: list) -> pd.DataFrame:
    """
    Kids sizes (Monate / Jahre / schmal variants).
    Frühchen always first. Sizes converted to months for sorting.
    Inconsistencies like '6Jahre' auto-fixed to '6 Jahre'.
    """
    unique  = _clean_deduplicate(sizes, normalize_fn=_normalize_kids_size)
    ordered = sorted(unique, key=_kids_age_key)
    return _build_df(ordered, 'kids_age')


# ─────────────────────────────────────────────────────────────────────────────
# 9. watch_size  — mm numerically first, then alpha (S/M, M/L, L)
# ─────────────────────────────────────────────────────────────────────────────

_WATCH_ALPHA_ORDER = {'S/M': 1, 'M/L': 2, 'L': 3}

def _watch_size_key(size):
    s = str(size).strip()
    mm = re.search(r'(\d+)\s*mm', s, re.I)
    if mm:
        return (0, int(mm.group(1)))
    if s in _WATCH_ALPHA_ORDER:
        return (1, _WATCH_ALPHA_ORDER[s])
    return (999, 999)

def watch_size_sort(sizes: list) -> pd.DataFrame:
    """
    Watch sizes: numeric mm first (ascending), then alpha (S/M → M/L → L).
    """
    unique  = _clean_deduplicate(sizes)
    ordered = sorted(unique, key=_watch_size_key)
    return _build_df(ordered, 'watch_size')


# ─────────────────────────────────────────────────────────────────────────────
# 10. jewellery_size  — mm ring (0), cm chain (1), alpha (2)
# ─────────────────────────────────────────────────────────────────────────────

_US_RING_TO_MM = {
    3.0: 44.2, 3.5: 45.5, 4.0: 46.8, 4.5: 48.0,
    5.0: 49.3, 5.5: 50.6, 6.0: 51.9, 6.5: 53.1,
    7.0: 54.4, 7.5: 55.7, 8.0: 57.0, 8.5: 58.3,
    9.0: 59.5, 9.5: 60.8, 10.0: 62.1,
}
_JEWEL_ALPHA_ORDER = {'S-M': 1, 'M-L': 2, 'L-XL': 3}

def _normalize_jewellery_size(s):
    s = str(s).strip()
    s = re.sub(r'(\d)(cm)$', r'\1 cm', s, flags=re.I)
    s = re.sub(r'(\d)(mm)$', r'\1 mm', s, flags=re.I)
    return s

def _jewellery_key(size):
    s = _normalize_jewellery_size(str(size).strip())
    if s in _JEWEL_ALPHA_ORDER:
        return (2, _JEWEL_ALPHA_ORDER[s])
    mm = re.fullmatch(r'(\d+(?:\.\d+)?)\s*mm', s, re.I)
    if mm:
        return (0, float(mm.group(1)))
    cm = re.fullmatch(r'(\d+(?:\.\d+)?)\s*cm', s, re.I)
    if cm:
        return (1, float(cm.group(1)))
    us = re.fullmatch(r'\d+(?:\.\d+)?', s)
    if us:
        us_val  = float(s)
        mm_equiv = _US_RING_TO_MM.get(us_val, us_val * 8.0)
        return (0, mm_equiv)
    return (999, 999)

def jewellery_size_sort(sizes: list) -> pd.DataFrame:
    """
    Jewellery sizes: mm ring circumference → cm chain/bracelet length → alpha.
    US ring sizes (e.g. '6.5') converted to mm equivalent for sorting.
    Fixes missing space before unit ('17cm' → '17 cm').
    """
    unique  = _clean_deduplicate(sizes, normalize_fn=_normalize_jewellery_size)
    ordered = sorted(unique, key=_jewellery_key)
    return _build_df(ordered, 'jewellery_size')


# ─────────────────────────────────────────────────────────────────────────────
# 11. headwear_size  — cm exact/range → age → alpha/label
# ─────────────────────────────────────────────────────────────────────────────

_HEADWEAR_ALPHA_ORDER = {
    'XXS-XS': 1, 'S-M': 2, 'M-L': 3, 'L-XL': 4, 'XL-XXL': 5,
    'Kids': 6, 'Kindergröße': 7, 'Senior': 8,
}

def _normalize_headwear_size(s):
    s = str(s).strip()
    s = re.sub(r'(\d),(\d)', r'\1.\2', s)  # German decimal comma
    return s

def _headwear_key(size):
    s = _normalize_headwear_size(size)
    # exact cm
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*cm', s, re.I)
    if m:
        return (0, float(m.group(1)), 0.0)
    # cm range with unit
    m = re.fullmatch(r'(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*cm', s, re.I)
    if m:
        return (0, float(m.group(1)), float(m.group(2)))
    # bare numeric range (assumed cm)
    m = re.fullmatch(r'(\d+)-(\d+)', s)
    if m:
        return (0, float(m.group(1)), float(m.group(2)))
    # age-based
    unit = 1 if re.search(r'monat', s, re.I) else (12 if re.search(r'jahr', s, re.I) else None)
    if unit is not None:
        nums = re.findall(r'\d+', s)
        if nums:
            a = int(nums[0]); b = int(nums[1]) if len(nums) > 1 else a
            return (1, float(a * unit), float(b * unit))
    if s in _HEADWEAR_ALPHA_ORDER:
        return (2, float(_HEADWEAR_ALPHA_ORDER[s]), 0.0)
    return (999, 999.0, 999.0)

def headwear_size_sort(sizes: list) -> pd.DataFrame:
    """
    Headwear sizes: cm exact/range first → age-based → alpha labels.
    Fixes German decimal comma ('62,5 cm' → '62.5 cm').
    """
    unique  = _clean_deduplicate(sizes, normalize_fn=_normalize_headwear_size)
    ordered = sorted(unique, key=_headwear_key)
    return _build_df(ordered, 'headwear_size')


# ─────────────────────────────────────────────────────────────────────────────
# 12. belt_size  — exact cm → cm range → alpha
# ─────────────────────────────────────────────────────────────────────────────

_BELT_ALPHA_ORDER = {'XS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5}

def _belt_key(size):
    s = str(size).strip()
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*cm', s, re.I)
    if m:
        return (0, float(m.group(1)), 0.0)
    m = re.fullmatch(r'(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*cm', s, re.I)
    if m:
        return (1, float(m.group(1)), float(m.group(2)))
    if s in _BELT_ALPHA_ORDER:
        return (2, float(_BELT_ALPHA_ORDER[s]), 0.0)
    return (999, 999.0, 0.0)

def belt_size_sort(sizes: list) -> pd.DataFrame:
    """
    Belt sizes: exact cm ascending → cm range (by lower bound) → alpha.
    """
    unique  = _clean_deduplicate(sizes)
    ordered = sorted(unique, key=_belt_key)
    return _build_df(ordered, 'belt_size')


# ─────────────────────────────────────────────────────────────────────────────
# 13. baggage_size  — cm → litres → alpha
# ─────────────────────────────────────────────────────────────────────────────

_BAGGAGE_ALPHA_ORDER = {'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 5}

def _normalize_baggage_size(s):
    s = str(s).strip()
    s = re.sub(r'(\d),(\d)', r'\1.\2', s)          # German decimal comma
    s = re.sub(r'(\d)([lL])$', r'\1 l', s)         # '12l' → '12 l'
    s = re.sub(r'(\d+(?:\.\d+)?)\s+L$', r'\1 l', s)  # '12 L' → '12 l'
    return s

def _baggage_key(size):
    s = _normalize_baggage_size(size)
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*cm', s, re.I)
    if m:
        return (0, float(m.group(1)))
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*l', s)
    if m:
        return (1, float(m.group(1)))
    if s in _BAGGAGE_ALPHA_ORDER:
        return (2, float(_BAGGAGE_ALPHA_ORDER[s]))
    return (999, 999.0)

def baggage_size_sort(sizes: list) -> pd.DataFrame:
    """
    Baggage sizes: cm physical dimension → litre capacity → alpha.
    Fixes '21,5 l' (German comma) and '12l' (missing space).
    """
    unique  = _clean_deduplicate(sizes, normalize_fn=_normalize_baggage_size)
    ordered = sorted(unique, key=_baggage_key)
    return _build_df(ordered, 'baggage_size')


# ─────────────────────────────────────────────────────────────────────────────
# 14. eyewear_size  — (lens_width, bridge_width, temple_length)
# ─────────────────────────────────────────────────────────────────────────────

def _eyewear_key(size):
    s = str(size).strip()
    m = re.fullmatch(r'(\d+)/(\d+)/(\d+)', s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r'(\d+)-(\d+)', s)
    if m:
        return (int(m.group(1)), int(m.group(2)), 0)
    m = re.fullmatch(r'\d+', s)
    if m:
        return (int(s), 0, 0)
    return (999, 999, 999)

def eyewear_size_sort(sizes: list) -> pd.DataFrame:
    """
    Eyewear sizes with three formats:
      '53/20/140' → (lens_width / bridge_width / temple_length)
      '52-18'     → (lens_width / bridge_width)
      '55'        → lens_width only
    Sorted by (lens ASC, bridge ASC, temple ASC). Missing dims default to 0.
    """
    unique  = _clean_deduplicate(sizes)
    ordered = sorted(unique, key=_eyewear_key)
    return _build_df(ordered, 'eyewear_size')


# ─────────────────────────────────────────────────────────────────────────────
# 15. one_size  — only Einheitsgröße at its fixed absolute order
# ─────────────────────────────────────────────────────────────────────────────

def one_size_sort(sizes=None) -> pd.DataFrame:
    """
    one_size group: only entry is 'Einheitsgröße' at EINHEITSGROESSE_GT_ORDER.
    """
    return pd.DataFrame([{
        'gt_size': EINHEITSGROESSE_GT_SIZE,
        'gt_size_order': EINHEITSGROESSE_GT_ORDER,
        'size_grouping': EINHEITSGROESSE_GROUPING,
    }])


# ─────────────────────────────────────────────────────────────────────────────
# 16. shoes_number_uk  — UK shoe sizes: numeric → standard (XS-XXL) → cm
# ─────────────────────────────────────────────────────────────────────────────

_UK_SHOES_STANDARD_RANK = {
    "XX-SMALL": 1, "X-SMALL": 2, "XS": 2,
    "SMALL": 3, "S": 3,
    "MEDIUM": 4, "M": 4,
    "LARGE": 5, "L": 5,
    "X-LARGE": 6, "XL": 6,
    "XX-LARGE": 7, "XXL": 7,
    "3XL": 8, "4XL": 9, "5XL": 10,
}
_UK_SHOES_AGE_PRIORITY = {"Child": 0, "Adult": 1}
_UK_SHOES_STYLE_PRIORITY = {
    "Narrow": 0,
    "Gender": 1,   # Men/Women
    "Normal": 2,
    "Wide": 3,
    "X-Wide": 4,
}

def _uk_shoes_number_key(size):
    s = str(size).strip()

    # cm sizes (after standard sizes)
    if "cm" in s:
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        num = float(m.group(1)) if m else 999_999.0
        return (4, num, 0, 0, 0)

    # Standard sizes (XS, S, M, L, XL … optionally with a trailing number)
    for std, rank in _UK_SHOES_STANDARD_RANK.items():
        if re.match(rf'^{re.escape(std)}(\b|\s)', s):
            m = re.search(r'\d+(?:\.\d+)?', s)
            if m:
                return (3, rank, float(m.group()), 0, 0)
            return (2, rank, 0, 0, 0)

    # Numeric UK sizes, with age / width / gender tie-breaks
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    number = float(m.group(1)) if m else 999_999.0

    age = _UK_SHOES_AGE_PRIORITY["Child"] if "Child" in s else _UK_SHOES_AGE_PRIORITY["Adult"]

    if "Narrow" in s:
        style = _UK_SHOES_STYLE_PRIORITY["Narrow"]
    elif " Men" in s or " Women" in s:
        style = _UK_SHOES_STYLE_PRIORITY["Gender"]
    elif "X-Wide" in s:
        style = _UK_SHOES_STYLE_PRIORITY["X-Wide"]
    elif "Wide" in s:
        style = _UK_SHOES_STYLE_PRIORITY["Wide"]
    else:
        style = _UK_SHOES_STYLE_PRIORITY["Normal"]

    return (1, number, age, style, s)

def shoes_number_uk_sort(sizes: list) -> pd.DataFrame:
    """
    UK shoe sizes ('6 UK', '9.5 UK Child', '4 UK Men/ 5 UK Women', '8 UK Wide' …),
    standard letter sizes (XS-XXL), and cm sizes.
    Sorted by (numeric ASC, age: Child before Adult, style: Narrow < Gender < Normal < Wide < X-Wide);
    standard sizes sort before numeric UK sizes, cm sizes sort last.
    'One Size' is stripped from the source (the shared one_size sentinel is appended instead).
    """
    sizes = [s for s in sizes if str(s).strip().lower() != 'one size']
    unique  = _clean_deduplicate(sizes)
    ordered = sorted(unique, key=_uk_shoes_number_key)
    return _build_df(ordered, 'shoes_number_uk')


