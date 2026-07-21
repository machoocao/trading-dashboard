"""
Daily updater for the Trading Desk dashboard.

What it does:
1. Pulls live/last-close prices for SPX, NDX, DJI, RUT, VIX, GOLD, WTI, BTC via yfinance
   and rewrites the `ticks` array in index.html.
2. Computes a simple VIX-based fear/greed proxy number and updates the sentiment gauge.
3. Calls the Claude API (with web search enabled) to research today's actual market-moving
   news and generate the catalysts (bullish/bearish/watch) and macro/global sections.
4. Updates the status pill (open/closed) and footer snapshot timestamp.
5. Writes the result back to index.html.

Env vars required (set as GitHub Actions secrets):
  ANTHROPIC_API_KEY

Note: this is a starting point, not a finished trading system. The fear/greed number is a
simplified VIX-based proxy, not the official CNN Fear & Greed Index (which blends 7 signals).
The holiday list below is hand-maintained and should be updated yearly.
"""

import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import yfinance as yf
from bs4 import BeautifulSoup

ANTHROPIC_API_KEY = __import__("os").environ["ANTHROPIC_API_KEY"]
HTML_PATH = "index.html"
MODEL = "claude-sonnet-5"

# --- 1. Ticker data -----------------------------------------------------

TICKERS = [
    ("SPX", "^GSPC"),
    ("NDX", "^NDX"),
    ("DJI", "^DJI"),
    ("RUT", "^RUT"),
    ("VIX", "^VIX"),
    ("GOLD", "GC=F"),
    ("WTI", "CL=F"),
    ("BTC", "BTC-USD"),
]


def fmt_value(x: float) -> str:
    return f"{x:,.2f}"


def fetch_ticks():
    ticks = []
    vix_level = None
    for sym, yf_symbol in TICKERS:
        hist = yf.Ticker(yf_symbol).history(period="5d")
        if len(hist) < 2:
            continue
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        chg_pct = (last - prev) / prev * 100
        up = chg_pct >= 0
        ticks.append(
            {
                "sym": sym,
                "val": fmt_value(last),
                "chg": f"{chg_pct:+.2f}%",
                "up": up,
            }
        )
        if sym == "VIX":
            vix_level = last
    return ticks, vix_level


# --- 2. Fear/greed proxy (VIX-based, simplified) ------------------------

def vix_to_fear_score(vix: float) -> int:
    """Rough proxy: VIX ~12 -> greed (80), VIX ~35+ -> extreme fear (5). Not the real CNN index."""
    score = 100 - ((vix - 10) / (40 - 10)) * 100
    return max(0, min(100, round(score)))


def fear_label(score: int) -> str:
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score < 55:
        return "Neutral"
    if score < 75:
        return "Greed"
    return "Extreme Greed"


# --- 3. Market status ----------------------------------------------------

US_HOLIDAYS_2026 = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}


def market_status():
    now = datetime.now(ZoneInfo("America/New_York"))
    today_str = now.strftime("%Y-%m-%d")
    is_weekday = now.weekday() < 5
    is_holiday = today_str in US_HOLIDAYS_2026
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    is_open = is_weekday and not is_holiday and open_time <= now <= close_time
    return is_open, now


# --- 4. Claude API call for narrative content -----------------------------

NARRATIVE_PROMPT = """Research today's actual U.S. stock market news using web search, then respond with
ONLY a JSON object (no markdown fences, no preamble) matching this exact schema:

{
  "bias_tags": [{"label": "NDX: risk-off", "type": "bear"}, ...],   // exactly 3 tags, type is one of bull/bear/amber
  "sentiment_note": "2-3 sentence plain-text note on today's market tone",
  "bullish": [{"text": "...", "tag": "single name"}, ...],   // 3-4 items
  "bearish": [{"text": "...", "tag": "semis"}, ...],         // 2-3 items
  "watch": [{"text": "...", "tag": "macro"}, ...],           // 3-4 items
  "macro": [{"day": "TODAY", "headline": "Bold headline", "body": "1-2 sentence explanation"}, ...] // 4-5 items
}

Base every item on real news you find via search. Do not fabricate figures. Keep each text field concise
(under 30 words) in the style of a professional trading desk note."""


def fetch_narrative():
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": NARRATIVE_PROMPT}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text_blocks = [b["text"] for b in data["content"] if b.get("type") == "text"]
    full_text = "\n".join(text_blocks).strip()
    full_text = re.sub(r"^```json\s*|\s*```$", "", full_text.strip())
    return json.loads(full_text)


# --- 5. HTML rewriting -----------------------------------------------------

def update_html(ticks, vix_level, narrative):
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Status pill
    is_open, now = market_status()
    pill = soup.find(id="statusPill")
    if is_open:
        pill["class"] = ["status-pill", "open"]
        pill_text = f"Market Open — {now.strftime('%a %b %-d')}"
    else:
        pill["class"] = ["status-pill", "closed"]
        pill_text = f"Closed — {now.strftime('%a %b %-d')}"
    dot = pill.find("span", class_="dot")
    pill.clear()
    pill.append(dot)
    pill.append(pill_text)

    # Fear/greed gauge
    score = vix_to_fear_score(vix_level) if vix_level else 50
    label = fear_label(score)
    soup.find(id="fearNum").string = str(score)
    gauge_label = soup.find(id="gaugeLabel")
    gauge_label.clear()
    b_tag = soup.new_tag("b")
    b_tag.string = label
    gauge_label.append(b_tag)
    gauge_label.append(f" — VIX at {vix_level:.1f}." if vix_level else "")
    soup.find(id="gaugeNeedle")["style"] = f"left:{score}%;"

    # Bias tags
    bias_row = soup.find(id="biasRow")
    bias_row.clear()
    for tag in narrative.get("bias_tags", []):
        span = soup.new_tag("span", **{"class": f"bias-tag {tag['type']}"})
        span.string = tag["label"]
        bias_row.append(span)

    # Sentiment note
    soup.find(id="sentimentNote").string = narrative.get("sentiment_note", "")

    # Catalyst lists
    def fill_list(list_id, items):
        container = soup.find(id=list_id)
        container.clear()
        for item in items:
            div = soup.new_tag("div", **{"class": "cat-item"})
            div.append(item["text"] + " ")
            tag_span = soup.new_tag("span", **{"class": "tag"})
            tag_span.string = item.get("tag", "")
            div.append(tag_span)
            container.append(div)

    fill_list("bullList", narrative.get("bullish", []))
    fill_list("bearList", narrative.get("bearish", []))
    fill_list("amberList", narrative.get("watch", []))

    # Macro list
    macro_container = soup.find(id="macroList")
    macro_container.clear()
    for item in narrative.get("macro", []):
        row = soup.new_tag("div", **{"class": "macro-item"})
        day = soup.new_tag("div", **{"class": "macro-day"})
        day.string = item.get("day", "")
        body = soup.new_tag("div", **{"class": "macro-body"})
        p = soup.new_tag("p")
        b = soup.new_tag("b")
        b.string = item.get("headline", "")
        p.append(b)
        p.append(" " + item.get("body", ""))
        body.append(p)
        row.append(day)
        row.append(body)
        macro_container.append(row)

    # Footer snapshot
    soup.find(id="footerSnapshot").string = (
        f"Snapshot auto-updated {now.strftime('%a %b %-d, %Y')} ~{now.strftime('%-I:%M %p')} ET — "
        "prices move fast intraday, treat levels as directional not live."
    )

    html = str(soup)

    # Ticks array (regex swap between markers, since it lives inside a <script> tag)
    ticks_js = ",\n    ".join(
        f"{{sym:'{t['sym']}', val:'{t['val']}', chg:'{t['chg']}', up:{'true' if t['up'] else 'false'}}}"
        for t in ticks
    )
    new_block = (
        "// TICKS_START (do not edit this line — script replaces the array below)\n"
        "  const ticks = [\n    " + ticks_js + ",\n  ];\n"
        "  // TICKS_END (do not edit this line)"
    )
    html = re.sub(
        r"// TICKS_START.*?// TICKS_END \(do not edit this line\)",
        new_block,
        html,
        flags=re.DOTALL,
    )

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ticks, vix_level = fetch_ticks()
    narrative = fetch_narrative()
    update_html(ticks, vix_level, narrative)
    print("Dashboard updated successfully.")


if __name__ == "__main__":
    main()
