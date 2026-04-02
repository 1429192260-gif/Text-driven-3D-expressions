import re


DEGREE_WORDS = [
    "?", "??", "?", "?", "??", "??", "??", "??", "??", "??", "??",
]

NEGATION_WORDS = [
    "?", "?", "??", "?", "??", "??", "??", "???",
]

UNCERTAINTY_WORDS = [
    "??", "??", "??", "??", "??", "??", "?", "??", "???",
]

LAUGHTER_PATTERNS = [
    "??", "??", "??",
]

SAD_PATTERNS = [
    "??", "?", "??", "??",
]


def _count_substrings(text, patterns):
    return float(sum(text.count(pattern) for pattern in patterns))


def extract_semantic_features(text):
    text = text or ""
    char_count = len(text)
    exclamation_count = text.count("!") + text.count("?")
    question_count = text.count("?") + text.count("?")
    ellipsis_count = text.count("?")
    comma_pause_count = text.count("?") + text.count(",")
    repeat_punct_count = len(re.findall(r"[!???]{2,}", text))

    degree_count = _count_substrings(text, DEGREE_WORDS)
    negation_count = _count_substrings(text, NEGATION_WORDS)
    uncertainty_count = _count_substrings(text, UNCERTAINTY_WORDS)
    laughter_count = _count_substrings(text, LAUGHTER_PATTERNS)
    sad_cue_count = _count_substrings(text, SAD_PATTERNS)

    normalized_length = min(char_count / 30.0, 1.0)

    return [
        normalized_length,
        min(exclamation_count / 3.0, 1.0),
        min(question_count / 3.0, 1.0),
        min(ellipsis_count / 2.0, 1.0),
        min(comma_pause_count / 4.0, 1.0),
        min(repeat_punct_count / 2.0, 1.0),
        min(degree_count / 3.0, 1.0),
        min(negation_count / 3.0, 1.0),
        min(uncertainty_count / 3.0, 1.0),
        min(laughter_count / 3.0, 1.0),
        min(sad_cue_count / 3.0, 1.0),
    ]
