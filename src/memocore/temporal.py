"""Write-time temporal anchoring: resolve relative time expressions against the session date.

"I went there yesterday", said in a session dated 8 May 2023, is stored with
"yesterday = 7 May 2023", so neither the retriever nor the answer model has to do date arithmetic
later, and the raw text is kept untouched beside it. Rules only, no LLM: anchoring costs nothing per
turn and never produces a date the text does not imply.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import replace
from datetime import date, timedelta
from typing import Callable

from memocore.datasets.locomo import Conversation, Session

_MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): number for number, name in enumerate(calendar.month_abbr) if name})
_WEEKDAYS = {name.lower(): number for number, name in enumerate(calendar.day_name)}
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10}
_SESSION_DATE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
_WEEKDAY = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_SEASON_MONTHS = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}


def parse_session_date(text: str) -> date | None:
    """LoCoMo session stamps look like "1:56 pm on 8 May, 2023"."""
    match = _SESSION_DATE.search(text)
    if not match:
        return None
    month = _MONTHS.get(match.group(2).lower())
    if not month:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(1)))
    except ValueError:
        return None


def _day(d: date) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def _month(year: int, month: int) -> str:
    return f"{calendar.month_name[month]} {year}"


def _shift_month(d: date, months: int) -> tuple[int, int]:
    index = d.year * 12 + d.month - 1 + months
    return index // 12, index % 12 + 1


def _last_weekday(d: date, weekday: int) -> date:
    return d - timedelta(days=(d.weekday() - weekday) % 7 or 7)


def _next_weekday(d: date, weekday: int) -> date:
    return d + timedelta(days=(weekday - d.weekday()) % 7 or 7)


def _ago(match: re.Match, d: date) -> str:
    amount, unit = match.group(1).lower(), match.group(2).lower()
    n = _NUMBERS.get(amount) or (int(amount) if amount.isdigit() else None)
    if n is None:  # "a few", "several", "a couple of": keep it vague but dated
        return f"{amount} {unit}s before {_day(d)}"
    if unit == "day":
        return _day(d - timedelta(days=n))
    if unit == "week":
        return f"{amount} week{'s' if n > 1 else ''} before {_day(d)}"
    if unit == "month":
        return _month(*_shift_month(d, -n))
    return str(d.year - n)


def _season(match: re.Match, d: date) -> str:
    which, season = match.group(1).lower(), match.group(2).lower()
    start = _SEASON_MONTHS[season]
    latest = d.year if d.month >= start else d.year - 1  # year the most recent such season began
    running = (d.year - latest) * 12 + d.month - start < 3
    if which == "last":
        year = latest - 1 if running else latest
    elif which == "next":
        year = latest + 1
    else:  # "this": the one under way, otherwise the one ahead
        year = latest if running else latest + 1
    return f"{season} {year}"


Rule = tuple[re.Pattern, Callable[[re.Match, date], str]]

# Longer phrases come first; a span claimed by one rule is not matched again by a shorter one.
_RULES: list[Rule] = [
    (r"the day before yesterday", lambda m, d: _day(d - timedelta(days=2))),
    (r"yesterday|last night", lambda m, d: _day(d - timedelta(days=1))),
    (r"today|tonight|this (?:morning|afternoon|evening)", lambda m, d: _day(d)),
    (r"tomorrow", lambda m, d: _day(d + timedelta(days=1))),
    (r"(?:last|this past) weekend", lambda m, d: f"the weekend before {_day(d)}"),
    (r"(?:last|this past) week", lambda m, d: f"the week before {_day(d)}"),
    (r"this weekend", lambda m, d: f"the weekend of {_day(d)}"),
    (r"this week", lambda m, d: f"the week of {_day(d)}"),
    (r"next weekend", lambda m, d: f"the weekend after {_day(d)}"),
    (r"next week", lambda m, d: f"the week after {_day(d)}"),
    (rf"(?:last|this past) ({_WEEKDAY})", lambda m, d: (
        f"the {m.group(1).capitalize()} before {_day(d)} ({_day(_last_weekday(d, _WEEKDAYS[m.group(1).lower()]))})")),
    (rf"next ({_WEEKDAY})", lambda m, d: _day(_next_weekday(d, _WEEKDAYS[m.group(1).lower()]))),
    (r"last month", lambda m, d: _month(*_shift_month(d, -1))),
    (r"this month", lambda m, d: _month(d.year, d.month)),
    (r"next month", lambda m, d: _month(*_shift_month(d, 1))),
    (r"last year", lambda m, d: str(d.year - 1)),
    (r"this year", lambda m, d: str(d.year)),
    (r"next year", lambda m, d: str(d.year + 1)),
    (r"(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+|a few|a couple of|couple of|several) "
     r"(day|week|month|year)s? ago", _ago),
    (r"(last|this|next) (spring|summer|fall|autumn|winter)", _season),
    (r"the other day", lambda m, d: f"a few days before {_day(d)}"),
]
_COMPILED = [(re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE), resolve) for pattern, resolve in _RULES]


def anchor_text(text: str, said_on: date) -> list[tuple[str, str]]:
    """(expression as written, resolved time) for each relative time expression, in text order."""
    claimed: list[tuple[int, int]] = []
    found: list[tuple[int, str, str]] = []
    for pattern, resolve in _COMPILED:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < other_end and other_start < end for other_start, other_end in claimed):
                continue
            claimed.append((start, end))
            found.append((start, match.group(0), resolve(match, said_on)))
    return [(expression, resolved) for _, expression, resolved in sorted(found)]


def time_note(text: str, said_on: date) -> str | None:
    anchors = anchor_text(text, said_on)
    if not anchors:
        return None
    return "time: " + "; ".join(f"{expression} = {resolved}" for expression, resolved in anchors)


def anchor_conversation(conversation: Conversation) -> Conversation:
    """A copy of the conversation whose turns carry resolved-time notes; the raw text is unchanged."""
    sessions = []
    for session in conversation.sessions:
        said_on = parse_session_date(session.date_time)
        if said_on is None:
            sessions.append(session)
            continue
        turns = tuple(
            replace(turn, time_note=note) if (note := time_note(turn.text, said_on)) else turn
            for turn in session.turns
        )
        sessions.append(Session(session.index, session.date_time, turns))
    return replace(conversation, sessions=tuple(sessions))
