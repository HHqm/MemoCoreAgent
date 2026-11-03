"""What each API call costs, from DeepSeek's list prices, so every run reports its spend in yuan.

Prices are from https://api-docs.deepseek.com/quick_start/pricing as of 2026-10 (DeepSeek-V4.1-Flash-1017
and DeepSeek-V4-Pro-0813). A cache hit is about 50x cheaper than a miss, output costs four times a miss,
and off-peak hours cost half, so where a prompt puts its fixed text and when a batch runs both matter.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Yuan per million tokens at peak: input served from the prefix cache, input not cached, output.
PRICES_CNY: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.04, 2.0, 8.0),
    "deepseek-v4-pro": (0.30, 9.0, 27.0),
}
# Monday to Friday, UTC; 09:00-12:00 and 14:00-18:00 Beijing time. Chinese public holidays are off-peak
# too, which this table does not know, so a holiday run is priced at peak (an overestimate).
PEAK_HOURS_UTC = ((1, 4), (6, 10))
OFF_PEAK_FACTOR = 0.5


def is_peak(moment: datetime) -> bool:
    moment = moment.astimezone(timezone.utc)
    return moment.weekday() < 5 and any(start <= moment.hour < end for start, end in PEAK_HOURS_UTC)


def next_off_peak(moment: datetime) -> datetime:
    """The first minute at or after `moment` that is billed at the off-peak rate."""
    probe = moment.astimezone(timezone.utc).replace(second=0, microsecond=0)
    while is_peak(probe):
        probe = (probe + timedelta(hours=1)).replace(minute=0)
    return probe


def call_cost(
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    at: datetime | None = None,
) -> float | None:
    """Yuan for one call; None for a model without a known price. Without a time, peak is assumed."""
    prices = PRICES_CNY.get(model)
    if prices is None:
        return None
    hit, miss, output = prices
    cost = (cached_tokens * hit + (prompt_tokens - cached_tokens) * miss + completion_tokens * output) / 1_000_000
    return cost * (OFF_PEAK_FACTOR if at is not None and not is_peak(at) else 1.0)


# Typical tokens per call measured on the s300 runs (input, share of it cached, output including reasoning),
# used only to print an estimate before a run starts.
TYPICAL_CALL = {
    "answer": (1_700, 0.0, 300),
    "answer_full_context": (21_300, 0.96, 140),
    "judge": (360, 0.0, 360),
}
BEIJING = timezone(timedelta(hours=8))


def estimate_line(model: str, calls: int, kind: str, now: datetime | None = None) -> str:
    """One line for the console: the expected cost of `calls` calls now, and when it would be half price."""
    now = now or datetime.now(timezone.utc)
    prompt, cached_share, output = TYPICAL_CALL[kind]
    per_call = call_cost(model, prompt, int(prompt * cached_share), output, now)
    if per_call is None or calls == 0:
        return ""
    line = f"预计花费约 ¥{per_call * calls:.2f}（按以往每次约 {prompt:,} 输入 / {output} 输出 token 估算）"
    if is_peak(now):
        cheap = next_off_peak(now).astimezone(BEIJING)
        line += f"；现在是高峰时段，北京时间 {cheap:%m-%d %H:%M} 起半价"
    else:
        line += "；现在是低谷时段（半价）"
    return line
