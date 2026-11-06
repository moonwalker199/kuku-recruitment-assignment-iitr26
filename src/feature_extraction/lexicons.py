
"""
src/feature_extraction/lexicons.py

Curated delivery-tag and promotional keyword dictionaries.

The vocabulary is based on corpus analysis of 50 Kuku FM episode
scripts. Matching is deterministic and interpretable.

Design principles:
- Preserve the curated delivery-tag vocabulary.
- Normalize punctuation and whitespace before matching.
- Resolve overlapping tag categories using the strongest signal.
- Avoid unrestricted substring matching.
- Use word-boundary-aware keyword matching.
"""

from __future__ import annotations

import re
import unicodedata


# ===========================================================================
# Text normalization and matching helpers
# ===========================================================================

def normalize_text(text: str) -> str:
    """Normalize text for consistent, case-insensitive matching."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_phrase(text: str, phrase: str) -> bool:
    """
    Check whether a phrase occurs as complete words.

    Prevents false positives such as:
    - 'die' matching 'died'
    - 'lie' matching 'believe'
    - 'over' matching 'lover'

    Multi-word phrases are matched as consecutive words.
    """
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)

    if not normalized_text or not normalized_phrase:
        return False

    pattern = r"(?<!\w)" + re.escape(normalized_phrase) + r"(?!\w)"
    return re.search(pattern, normalized_text) is not None


# ===========================================================================
# Delivery-tag taxonomic groups
# ===========================================================================

HIGH_CONFLICT_TAGS: frozenset[str] = frozenset({
    "furious", "screaming", "livid", "roaring", "enraged", "seething",
    "threatening", "menacing", "vicious", "choking with rage", "incensed",
    "bellowing", "ferocious", "thundering", "thunderous", "violent",
    "brutal", "fierce", "shouting", "yelling", "barking", "barked",
    "screaming command", "enraged scream", "roaring defiance",
    "roaring in terror", "roaring with conviction", "roaring in torment",
    "pained fury", "simmering with fury", "cold fury", "shouted",
    "shrieking", "screeching", "shrill", "thunderous finale",
})

DISTRESS_TAGS: frozenset[str] = frozenset({
    "desperate", "trembling", "heartbroken", "sobbing", "wailing",
    "devastated", "pleading", "horrified", "anguished", "distraught",
    "crying", "crying out", "weeping", "quietly weeping",
    "choking on tears", "choked sob", "choking", "gasping",
    "gasping cry", "gasping in pain", "grief-stricken", "mournful",
    "sorrowful", "broken", "shattered", "despairing", "despondent",
    "panicked", "panic-stricken", "frantic", "hysterical",
    "hysterically sobbing", "terrified", "frightened", "fearful",
    "alarmed", "distressed", "agonized", "agonized scream",
    "pain-filled", "pained", "stricken", "begging", "pleading",
    "desperate shout", "desperately defending", "defiant gasps",
    "voice breaking", "voice continuing",
})

DEFIANCE_TAGS: frozenset[str] = frozenset({
    "defiant", "resolute", "unwavering", "fierce", "indomitable",
    "roaring defiance", "passionate defiance", "unyielding",
    "uncompromising", "unbreakable", "unbowed", "unintimidated",
    "unflinching", "firm", "steely", "steely resolve",
    "bitterly resolute", "quietly resolute", "fierce resolve",
    "indomitable spirit", "resilient", "wildly defiant",
    "softly defiant", "roaring with conviction", "unstoppable",
    "unstoppable vow", "roaring defiance", "passionate",
    "passionate defiance", "triumphant in spirit",
})

CLIFFHANGER_TAGS: frozenset[str] = frozenset({
    "ominous", "chilling", "fateful", "vow of vengeance", "solemn vow",
    "thunderous finale", "unstoppable vow", "deadly", "lethal",
    "menacing promise", "trap snapping", "chillingly calm",
    "chillingly detached", "chilling proclamation", "dark", "sinister",
    "dangerous", "deadly serious", "lethally cold",
    "venomous monologue", "venomous satisfaction",
    "coldly severing", "menacing snarl", "quietly venomous",
    "poisonous", "predatory", "calculated",
})

REVELATION_TAGS: frozenset[str] = frozenset({
    "startled", "aghast", "astounded", "shocked", "horrified",
    "realizing", "revelatory", "in disbelief", "shellshocked",
    "stunned", "bewildered", "astonished", "perplexed", "baffled",
    "disbelieving", "unbelieving", "incredulous", "gasping",
    "startled whisper",
})

MOCKERY_TAGS: frozenset[str] = frozenset({
    "mocking", "smug", "gloating", "condescending", "disdainful",
    "dismissive", "cruel", "callous", "scornful", "contemptuous",
    "derisive", "sarcastic", "sardonic", "sneering", "snide",
    "taunting", "patronizing", "mock-sympathetic",
    "mockingly reverent", "sadistic", "viciously calm",
    "viciously delighted",
})

COLD_CALCULATING_TAGS: frozenset[str] = frozenset({
    "cold", "icy", "hard", "steely", "measured",
    "calculating", "scheming", "conniving", "devious",
    "deceptive", "ruthless", "implacable", "inexorable",
    "coldly curious", "coldly dismissive", "coldly inquisitive",
})


# ===========================================================================
# Delivery-tag group weights
# ===========================================================================

DELIVERY_GROUP_WEIGHTS: dict[str, float] = {
    "CLIFFHANGER": 1.00,
    "HIGH_CONFLICT": 0.95,
    "DISTRESS": 0.85,
    "DEFIANCE": 0.80,
    "REVELATION": 0.75,
    "MOCKERY": 0.70,
    "COLD_CALCULATING": 0.60,
}

DELIVERY_TAG_GROUPS: dict[str, frozenset[str]] = {
    "CLIFFHANGER": CLIFFHANGER_TAGS,
    "HIGH_CONFLICT": HIGH_CONFLICT_TAGS,
    "DISTRESS": DISTRESS_TAGS,
    "DEFIANCE": DEFIANCE_TAGS,
    "REVELATION": REVELATION_TAGS,
    "MOCKERY": MOCKERY_TAGS,
    "COLD_CALCULATING": COLD_CALCULATING_TAGS,
}

# Pre-normalize the vocabulary once instead of doing it on every lookup.
_NORMALIZED_DELIVERY_GROUPS: dict[str, frozenset[str]] = {
    group: frozenset(normalize_text(tag) for tag in members)
    for group, members in DELIVERY_TAG_GROUPS.items()
}


# ===========================================================================
# Conflict-class tags
# ===========================================================================

CONFLICT_TAGS: frozenset[str] = frozenset(
    normalize_text(tag)
    for tag in (
        HIGH_CONFLICT_TAGS
        | MOCKERY_TAGS
        | frozenset({
            "accusatory", "accusing", "condemning", "condemnatory",
            "confrontational", "aggressive", "dominant", "imperious",
            "belligerent", "bitter", "hateful", "spiteful", "vindictive",
            "venomous", "choking with rage", "incensed", "livid",
        })
    )
)


# ===========================================================================
# Cliffhanger keywords
# ===========================================================================
#
# These are searched in ACTION and SOUND FX descriptions.
# Generic words are retained only when they represent a useful sound,
# atmosphere or dramatic transition.
# ===========================================================================

CLIFFHANGER_KEYWORDS: tuple[str, ...] = (
    "ominous", "cliffhanger", "drone", "swells", "peak",
    "cut to black", "silence", "fade to black", "fade out",
    "thunderclap", "gunshot", "scream", "gasp", "crash",
    "explosion", "darkness", "shadows", "footsteps",
    "door slams", "door slam",
)


# ===========================================================================
# Promotional hook keywords
# ===========================================================================
#
# These are searched in DIALOGUE and ACTION descriptions.
# Prefer meaningful phrases and dramatic vocabulary over generic words.
# ===========================================================================

HOOK_KEYWORDS: tuple[str, ...] = (
    # Betrayal and deception
    "betray", "betrayed", "betrayal", "swore", "promised",
    "lied to me", "deceive", "deceived", "knew all along",

    # Life-and-death stakes
    "die", "dead", "death", "kill", "murder", "sentence",
    "execution", "hang", "executed", "assassin", "poison",

    # Power and authority
    "duke", "throne", "rightful", "usurper", "claim",
    "witchcraft", "arrest", "imposter", "proclaimed",
    "rightful heir",

    # Revelation and hidden truth
    "truth", "secret", "discovered", "exposed",
    "unmasked", "framed", "reveal", "confession",

    # Emotional turning points
    "enough is enough", "left me", "abandoned",
    "walked away", "never again",

    # Direct confrontation
    "how dare", "i will not", "you cannot",
    "you will not", "i have come", "here i stand",
    "we settle this", "choose", "leave me",
    "i chose",
)


# ===========================================================================
# Protagonist speaker names
# ===========================================================================

PROTAGONIST_NAMES: frozenset[str] = frozenset({
    "juliana",
    "lady juliana",
    "alexander",
    "mysterious bearded man",
    "scarred bearded man",
})

NORMALIZED_PROTAGONIST_NAMES: frozenset[str] = frozenset(
    normalize_text(name) for name in PROTAGONIST_NAMES
)


# ===========================================================================
# Delivery-tag scoring
# ===========================================================================

def score_delivery_tag(tag: str) -> float:
    """
    Return the promotional intensity score for a delivery tag.

    Matching policy:
    1. Normalize the input.
    2. Return the strongest exact category match.
    3. For compound tags, allow conservative phrase containment.
    4. Return 0.0 for unknown tags.

    Multiple category matches are resolved using the maximum group weight.
    This prevents dictionary ordering from changing the score.

    Unknown tags receive zero because an unrecognized delivery direction
    should not automatically be interpreted as promotional intensity.
    """
    normalized = normalize_text(tag)

    if not normalized:
        return 0.0

    exact_scores = [
        DELIVERY_GROUP_WEIGHTS[group]
        for group, members in _NORMALIZED_DELIVERY_GROUPS.items()
        if normalized in members
    ]

    if exact_scores:
        return max(exact_scores)

    # Conservative compound-tag matching.
    # Only match when the shorter complete phrase is contained in the
    # longer normalized phrase, and the shorter phrase has at least
    # two meaningful words or is a sufficiently distinctive single word.
    partial_scores: list[float] = []

    for group, members in _NORMALIZED_DELIVERY_GROUPS.items():
        for member in members:
            if member == normalized:
                continue

            shorter, longer = (
                (member, normalized)
                if len(member) <= len(normalized)
                else (normalized, member)
            )

            if not contains_phrase(longer, shorter):
                continue

            words = shorter.split()

            if len(words) >= 2 or len(shorter) >= 8:
                partial_scores.append(
                    DELIVERY_GROUP_WEIGHTS[group] * 0.80
                )

    if partial_scores:
        return max(partial_scores)

    return 0.0


def is_conflict_tag(tag: str) -> bool:
    """Return True when a delivery tag belongs to the conflict vocabulary."""
    return normalize_text(tag) in CONFLICT_TAGS


def is_protagonist(speaker: str) -> bool:
    """Return True for a known protagonist speaker alias."""
    return normalize_text(speaker) in NORMALIZED_PROTAGONIST_NAMES
