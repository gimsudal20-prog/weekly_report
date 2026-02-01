# -*- coding: utf-8 -*-
"""
주간 광고 보고서 자동 생성기 (EXE 빌드용)

사용법 (exe 기준):
1) exe와 같은 폴더에 input.txt 생성 후, 주간 원문 텍스트 붙여넣기
2) weekly_report.exe 실행
3) outputs 폴더에 업체별 보고서 txt 생성
4) ad_weekly_history.sqlite 파일이 누적 저장되어 전주 대비(WoW) 비교 가능

input.txt 상단(선택) 메타 예시:
report_url: https://...
week_start: 2026-01-19
date_from: 2026-01-19
date_to: 2026-01-25
auto_mode: lastmonday   # (week_start/date_from~date_to 미입력 시) monday or lastmonday
"""

import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ========================
# 0) 상수 / 설정
# ========================

SUPPORTED_CHANNELS = [
    "네이버 SA",
    "네이버 SSA",
    "네이버 GFA",
    "구글 검색광고",
    "구글 GDN",
    "메타 배너광고",
]

CLIENT_CHANNEL_MAP = {
    "시종도어": ["네이버 SA", "네이버 SSA", "네이버 GFA"],
    "시공프로": ["네이버 SA", "네이버 SSA"],
    "HSW": ["네이버 SA", "네이버 SSA"],
    "실리콘플러스": ["네이버 SA", "네이버 SSA"],
    "조인스페이": ["네이버 SA", "구글 검색광고"],
    "휴비즈넷": ["네이버 SA"],
    "센터큐": ["네이버 SA", "메타 배너광고"],
    "스카이랩": ["네이버 SA"],
    "더마드라이": ["네이버 SA", "네이버 SSA", "네이버 GFA"],
    "비에스대우글": ["네이버 SA", "네이버 SSA"],
    "욥": ["네이버 SA"],
    "최씨본가": ["네이버 SA"],
    "골드앤": ["네이버 SA"],
    "포렌식스": ["네이버 SA"],
}

CHANNEL_DISPLAY = {
    "네이버 SA": "네이버 SA (파워링크)",
    "네이버 SSA": "네이버 SSA (쇼핑검색)",
    "네이버 GFA": "네이버 GFA",
    "구글 GDN": "구글 GDN",
    "구글 검색광고": "구글 검색광고",
    "메타 배너광고": "메타 배너광고",
}

CHANNEL_OUTPUT_ORDER = [
    "네이버 SA",
    "네이버 SSA",
    "네이버 GFA",
    "구글 GDN",
    "구글 검색광고",
    "메타 배너광고",
]

# 헤더 표준화 매핑 (normalize_header_key 후 매칭)
HEADER_TO_FIELD = {
    "노출수": "impressions",
    "클릭수": "clicks",
    "전환수": "conversions",
    "전환당비용원": "cpa",
    "클릭률": "ctr_pct",
    "클릭률퍼센트": "ctr_pct",
    "평균클릭비용vat포함원": "cpc_vat",
    "하루예산": "daily_budget",
    "총비용vat포함원": "cost_vat",
    "전환매출액원": "revenue",
    "광고수익률": "roas_pct",
    "광고수익률퍼센트": "roas_pct",
}

# A 포맷 키:값 유사어 (키 정규화 후 매칭)
KV_SYNONYMS = {
    "노출": "impressions",
    "노출수": "impressions",
    "impressions": "impressions",
    "클릭": "clicks",
    "클릭수": "clicks",
    "clicks": "clicks",
    "전환": "conversions",
    "전환수": "conversions",
    "conversions": "conversions",
    "ctr": "ctr_pct",
    "클릭률": "ctr_pct",
    "cpc": "cpc_vat",
    "평균클릭비용": "cpc_vat",
    "vat포함cpc": "cpc_vat",
    "cpa": "cpa",
    "전환당비용": "cpa",
    "비용": "cost_vat",
    "총비용": "cost_vat",
    "광고비": "cost_vat",
    "매출": "revenue",
    "전환매출": "revenue",
    "roas": "roas_pct",
    "광고수익률": "roas_pct",
    "하루예산": "daily_budget",
    "일예산": "daily_budget",
}

STANDARD_FIELDS = [
    "impressions", "clicks", "conversions",
    "ctr_pct", "cpc_vat", "cpa", "daily_budget",
    "cost_vat", "revenue", "roas_pct"
]


# ========================
# 1) 데이터 모델
# ========================

@dataclass
class Block:
    client: str
    channel: str
    body_text: str

@dataclass
class Record:
    client: str
    channel: str
    week_start: date
    week_end: date
    item: str
    metrics: Dict[str, Any]
    raw_text: str
    parse_warnings: List[str]

@dataclass
class EnrichedRecord(Record):
    prev_metrics: Optional[Dict[str, Any]] = None
    wow: Optional[Dict[str, Dict[str, Any]]] = None


# ========================
# 2) 유틸
# ========================

def parse_date_any(s: str) -> Optional[date]:
    s = s.strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None

def infer_week_range(
    week_start: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    auto_mode: str = "lastmonday",
) -> Tuple[date, date]:
    """
    - week_start 있으면 week_start~week_start+6
    - date_from/date_to 있으면 그대로
    - 아무것도 없으면:
      auto_mode="monday" => 이번 주(월~일)
      auto_mode="lastmonday" => 지난 주(월~일) (기본)
    """
    if week_start:
        ws = parse_date_any(week_start)
        if not ws:
            raise ValueError(f"week_start 파싱 실패: {week_start}")
        return ws, ws + timedelta(days=6)

    if date_from and date_to:
        df = parse_date_any(date_from)
        dt = parse_date_any(date_to)
        if not df or not dt:
            raise ValueError(f"date_from/date_to 파싱 실패: {date_from}~{date_to}")
        return df, dt

    today = date.today()
    ws = today - timedelta(days=today.weekday())  # 이번 주 월요일
    if auto_mode == "lastmonday":
        ws = ws - timedelta(days=7)
    return ws, ws + timedelta(days=6)

def normalize_header_key(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("(", "").replace(")", "")
    s = s.replace("%", "퍼센트")
    s = re.sub(r"[^\w가-힣]", "", s)
    return s

def normalize_channel_name(ch: str) -> str:
    raw = (ch or "").strip()
    slim = re.sub(r"\s+", "", raw)

    if slim in ("네이버SA", "SA", "파워링크", "네이버파워링크"):
        return "네이버 SA"
    if slim in ("네이버SSA", "SSA", "쇼핑검색", "네이버쇼핑검색"):
        return "네이버 SSA"
    if slim in ("네이버GFA", "GFA"):
        return "네이버 GFA"
    if slim in ("구글검색", "구글검색광고", "GoogleSearch", "Google검색"):
        return "구글 검색광고"
    if slim in ("구글GDN", "GDN"):
        return "구글 GDN"
    if slim in ("메타", "메타배너", "메타배너광고", "Meta"):
        return "메타 배너광고"

    return raw

def format_int(n: Optional[float]) -> str:
    if n is None:
        return "-"
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return "-"

def format_float(n: Optional[float], digits: int = 2) -> str:
    if n is None:
        return "-"
    try:
        return f"{float(n):.{digits}f}"
    except Exception:
        return "-"

def parse_number(token: Any) -> Optional[float]:
    if token is None:
        return None
    t = str(token).strip()
    if t == "" or t == "-" or t.lower() == "null":
        return None
    t = t.replace(",", "")
    t = re.sub(r"(회|건|원|%|퍼센트)", "", t)
    t = t.replace(" ", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
    if not m:
        return None
    num = m.group(0)
    try:
        if "." in num:
            return float(num)
        return float(int(num))
    except Exception:
        return None

def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        b = float(b)
        if b == 0.0:
            return None
        return float(a) / b
    except Exception:
        return None

def pct_change(prev: Optional[float], now: Optional[float]) -> Optional[float]:
    if prev is None or now is None:
        return None
    try:
        prev = float(prev)
        if prev == 0.0:
            return None
        return (float(now) - prev) / prev * 100.0
    except Exception:
        return None

def normalize_kv_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("(", "").replace(")", "")
    s = s.replace("%", "")
    s = s.replace(":", "").replace("：", "")
    s = re.sub(r"[^a-z0-9가-힣_]", "", s)
    return s


# ========================
# 3) split_blocks
# ========================

BLOCK_START_RE = re.compile(r"^\[(.+?)\]\[(.+?)\]\s*$")

def split_blocks(text: str) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []

    cur_client = None
    cur_channel = None
    cur_body: List[str] = []

    def flush():
        nonlocal cur_client, cur_channel, cur_body
        if cur_client and cur_channel:
            body_text = "\n".join(cur_body).strip("\n")
            blocks.append(Block(cur_client.strip(), normalize_channel_name(cur_channel.strip()), body_text))
        cur_client, cur_channel, cur_body = None, None, []

    for line in lines:
        m = BLOCK_START_RE.match(line.strip())
        if m:
            flush()
            cur_client, cur_channel = m.group(1), m.group(2)
        else:
            if cur_client is not None:
                cur_body.append(line.rstrip("\n"))
            else:
                continue

    flush()
    return blocks


# ========================
# 4) parse_block (B 우선, A 보조)
# ========================

def detect_header_mode(body_text: str) -> Tuple[Optional[List[str]], Optional[str]]:
    lines = [ln.strip() for ln in body_text.splitlines()]
    for i in range(len(lines)):
        if not lines[i]:
            continue
        headers: List[str] = []
        j = i
        while j < len(lines) and lines[j]:
            hk = normalize_header_key(lines[j])
            if hk in HEADER_TO_FIELD:
                headers.append(lines[j])
                j += 1
            else:
                break

        if len(headers) >= 6:
            for k in range(j, len(lines)):
                if not lines[k]:
                    continue
                if ("캠페인" in lines[k]) or ("결과" in lines[k]):
                    return headers, lines[k]
            return None, None
    return None, None

def parse_header_result(headers: List[str], result_line: str) -> Tuple[str, Dict[str, Any], List[str]]:
    warnings: List[str] = []
    std_fields: List[Optional[str]] = []

    for h in headers:
        hk = normalize_header_key(h)
        std_fields.append(HEADER_TO_FIELD.get(hk))

    # 2칸 이상 공백/탭 기준 split
    parts = re.split(r"\s{2,}|\t+", result_line.strip())
    parts = [p for p in parts if p != ""]
    if len(parts) < 2:
        parts = result_line.strip().split()
        if len(parts) < 2:
            return "채널합", {k: None for k in STANDARD_FIELDS}, ["B포맷 결과 라인 토큰화 실패"]

    item_label = parts[0].strip()
    value_tokens = parts[1:]

    metrics: Dict[str, Any] = {k: None for k in STANDARD_FIELDS}

    if len(value_tokens) == len(std_fields):
        for f, tok in zip(std_fields, value_tokens):
            if f:
                metrics[f] = parse_number(tok)
        return item_label, metrics, warnings

    if len(value_tokens) == len(std_fields) - 1 and "daily_budget" in std_fields:
        idx = std_fields.index("daily_budget")
        vt = value_tokens[:idx] + [None] + value_tokens[idx:]
        for f, tok in zip(std_fields, vt):
            if f:
                metrics[f] = parse_number(tok) if tok is not None else None
        warnings.append("daily_budget 값 누락으로 NULL 처리")
        return item_label, metrics, warnings

    warnings.append(f"헤더({len(std_fields)})-값({len(value_tokens)}) 개수 불일치")
    for f, tok in zip(std_fields, value_tokens):
        if f:
            metrics[f] = parse_number(tok)
    return item_label, metrics, warnings

def parse_key_value(body_text: str) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    metrics: Dict[str, Any] = {k: None for k in STANDARD_FIELDS}

    for ln in body_text.splitlines():
        line = ln.strip()
        if not line:
            continue
        if ":" not in line and "：" not in line:
            continue
        key, val = re.split(r"[:：]", line, maxsplit=1)
        nk = normalize_kv_key(key)
        field = KV_SYNONYMS.get(nk)
        if not field:
            for k2, f2 in KV_SYNONYMS.items():
                if k2 in nk:
                    field = f2
                    break
        if not field:
            continue
        metrics[field] = parse_number(val)

    if all(metrics[k] is None for k in STANDARD_FIELDS):
        warnings.append("A포맷 키:값 파싱 실패(매칭된 지표 없음)")
    return metrics, warnings

def normalize_metrics(metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    nm = {k: metrics.get(k, None) for k in STANDARD_FIELDS}

    imp = nm.get("impressions")
    clk = nm.get("clicks")
    conv = nm.get("conversions")
    cost = nm.get("cost_vat")
    rev = nm.get("revenue")

    ctr_calc = safe_div(clk, imp)
    ctr_calc = ctr_calc * 100 if ctr_calc is not None else None
    cpc_calc = safe_div(cost, clk)
    cpa_calc = safe_div(cost, conv)
    roas_calc = safe_div(rev, cost)
    roas_calc = roas_calc * 100 if roas_calc is not None else None

    def flag_if_diff(field: str, calc: Optional[float]):
        shown = nm.get(field)
        if shown is None or calc is None:
            return
        dp = pct_change(shown, calc)
        if dp is None:
            return
        if abs(dp) >= 10.0:
            warnings.append("표시 지표와 계산 지표 간 차이가 있어 집계 기준(부가세/전환정의) 점검 필요")

    flag_if_diff("ctr_pct", ctr_calc)
    flag_if_diff("cpc_vat", cpc_calc)
    flag_if_diff("cpa", cpa_calc)
    flag_if_diff("roas_pct", roas_calc)

    return nm, warnings

def parse_block(block: Block, week_start: date, week_end: date) -> List[Record]:
    warnings: List[str] = []

    if block.channel not in SUPPORTED_CHANNELS:
        warnings.append(f"지원하지 않는 채널명: {block.channel}")

    if block.client in CLIENT_CHANNEL_MAP:
        if block.channel not in CLIENT_CHANNEL_MAP[block.client]:
            warnings.append("업체별 활성 채널 맵과 불일치(입력 채널 확인 필요)")
    else:
        warnings.append("등록되지 않은 업체명(채널 맵 미존재)")

    headers, result_line = detect_header_mode(block.body_text)
    if headers and result_line:
        item_label, metrics, w2 = parse_header_result(headers, result_line)
        warnings.extend(w2)
        nm, w3 = normalize_metrics(metrics)
        warnings.extend(w3)
        return [Record(
            client=block.client,
            channel=block.channel,
            week_start=week_start,
            week_end=week_end,
            item=item_label or "채널합",
            metrics=nm,
            raw_text=f"[{block.client}][{block.channel}]\n{block.body_text}",
            parse_warnings=warnings[:],
        )]

    metrics, w2 = parse_key_value(block.body_text)
    warnings.extend(w2)
    nm, w3 = normalize_metrics(metrics)
    warnings.extend(w3)
    return [Record(
        client=block.client,
        channel=block.channel,
        week_start=week_start,
        week_end=week_end,
        item="채널합",
        metrics=nm,
        raw_text=f"[{block.client}][{block.channel}]\n{block.body_text}",
        parse_warnings=warnings[:],
    )]


# ========================
# 5) HISTORY 저장소 (SQLite)
# ========================

class HistoryStore:
    def __init__(self, db_path: str = "ad_weekly_history.sqlite"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            client TEXT NOT NULL,
            channel TEXT NOT NULL,
            week_start TEXT NOT NULL,
            week_end TEXT NOT NULL,
            item TEXT NOT NULL,
            impressions REAL,
            clicks REAL,
            conversions REAL,
            ctr_pct REAL,
            cpc_vat REAL,
            cpa REAL,
            daily_budget REAL,
            cost_vat REAL,
            revenue REAL,
            roas_pct REAL,
            raw_text TEXT,
            parse_warnings TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (client, channel, week_start, item)
        )
        """)
        self.conn.commit()

    def upsert_history(self, records: List[Record]):
        cur = self.conn.cursor()
        now_iso = datetime.now().isoformat(timespec="seconds")
        for r in records:
            row = {
                "client": r.client,
                "channel": r.channel,
                "week_start": r.week_start.isoformat(),
                "week_end": r.week_end.isoformat(),
                "item": r.item,
                **{k: r.metrics.get(k) for k in STANDARD_FIELDS},
                "raw_text": r.raw_text,
                "parse_warnings": "\n".join(r.parse_warnings),
                "updated_at": now_iso,
            }
            cur.execute("""
            INSERT INTO history (
                client, channel, week_start, week_end, item,
                impressions, clicks, conversions, ctr_pct, cpc_vat, cpa, daily_budget,
                cost_vat, revenue, roas_pct,
                raw_text, parse_warnings, updated_at
            )
            VALUES (
                :client, :channel, :week_start, :week_end, :item,
                :impressions, :clicks, :conversions, :ctr_pct, :cpc_vat, :cpa, :daily_budget,
                :cost_vat, :revenue, :roas_pct,
                :raw_text, :parse_warnings, :updated_at
            )
            ON CONFLICT(client, channel, week_start, item) DO UPDATE SET
                week_end=excluded.week_end,
                impressions=excluded.impressions,
                clicks=excluded.clicks,
                conversions=excluded.conversions,
                ctr_pct=excluded.ctr_pct,
                cpc_vat=excluded.cpc_vat,
                cpa=excluded.cpa,
                daily_budget=excluded.daily_budget,
                cost_vat=excluded.cost_vat,
                revenue=excluded.revenue,
                roas_pct=excluded.roas_pct,
                raw_text=excluded.raw_text,
                parse_warnings=excluded.parse_warnings,
                updated_at=excluded.updated_at
            """, row)
        self.conn.commit()

    def get_prev(self, client: str, channel: str, item: str, prev_week_start: date) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("""
        SELECT * FROM history
        WHERE client=? AND channel=? AND item=? AND week_start=?
        """, (client, channel, item, prev_week_start.isoformat()))
        row = cur.fetchone()
        if not row:
            return None
        return {k: row[k] for k in STANDARD_FIELDS}

    def close(self):
        self.conn.close()


# ========================
# 6) build_wow
# ========================

def build_wow(records: List[Record], store: HistoryStore) -> List[EnrichedRecord]:
    enriched: List[EnrichedRecord] = []
    for r in records:
        prev_week_start = r.week_start - timedelta(days=7)
        prev = store.get_prev(r.client, r.channel, r.item, prev_week_start)

        wow: Dict[str, Dict[str, Any]] = {}
        warnings = r.parse_warnings[:]

        if prev:
            for f in STANDARD_FIELDS:
                p = prev.get(f)
                n = r.metrics.get(f)
                diff = None if (p is None or n is None) else (float(n) - float(p))
                diff_pct = pct_change(p, n)
                wow[f] = {"prev": p, "now": n, "diff": diff, "diff_pct": diff_pct}

            conv_prev = prev.get("conversions")
            conv_now = r.metrics.get("conversions")
            rev_prev = prev.get("revenue")
            rev_now = r.metrics.get("revenue")

            conv_diff_pct = pct_change(conv_prev, conv_now)
            if conv_diff_pct is not None and (conv_diff_pct <= -70.0 or conv_diff_pct >= 200.0):
                warnings.append("전환수 변동폭이 커 집계/정의 점검 필요")

            rev_diff_pct = pct_change(rev_prev, rev_now)
            if rev_diff_pct is not None and abs(rev_diff_pct) >= 70.0:
                if conv_diff_pct is not None:
                    if (rev_diff_pct > 0 and conv_diff_pct < 0) or (rev_diff_pct < 0 and conv_diff_pct > 0):
                        warnings.append("매출/전환 흐름이 엇갈려 집계/정의 점검 필요")

        enriched.append(EnrichedRecord(
            client=r.client,
            channel=r.channel,
            week_start=r.week_start,
            week_end=r.week_end,
            item=r.item,
            metrics=r.metrics,
            raw_text=r.raw_text,
            parse_warnings=warnings,
            prev_metrics=prev,
            wow=wow if prev else None,
        ))
    return enriched


# ========================
# 7) render_report
# ========================

def _arrow(prev: Optional[float], now: Optional[float], kind: str) -> str:
    if prev is None or now is None:
        return "전주 - → 금주 -"
    if kind in ("int", "money"):
        p = format_int(prev)
        n = format_int(now)
        d = now - prev
        sign = "+" if d > 0 else ""
        return f"전주 {p} → 금주 {n} ({sign}{format_int(d)})"
    p = format_float(prev, 2)
    n = format_float(now, 2)
    d = now - prev
    sign = "+" if d > 0 else ""
    return f"전주 {p}% → 금주 {n}% ({sign}{format_float(d, 2)}%p)"

def _needs_check(warnings: List[str]) -> bool:
    key_phrases = ["점검 필요", "집계 기준", "정의 점검", "변동폭"]
    return any(any(k in w for k in key_phrases) for w in warnings)

def _comment_lines(er: EnrichedRecord) -> List[str]:
    ch = er.channel
    prev_exists = er.prev_metrics is not None
    w = er.wow or {}

    def get(field: str) -> Tuple[Optional[float], Optional[float]]:
        if not prev_exists:
            return None, None
        return (w.get(field, {}).get("prev"), w.get(field, {}).get("now"))

    lines: List[str] = []

    if prev_exists:
        roas_dp = w.get("roas_pct", {}).get("diff_pct")
        if roas_dp is None:
            lines.append(f"이번 주 {ch}는 전주 대비 지표 변동이 확인됩니다.")
        else:
            direction = "상승" if roas_dp > 0 else "하락" if roas_dp < 0 else "유지"
            lines.append(f"이번 주 {ch}는 전주 대비 ROAS가 {direction}한 흐름입니다.")
        lines.append(f"- 노출수: {_arrow(*get('impressions'), 'int')}")
        lines.append(f"- 클릭수: {_arrow(*get('clicks'), 'int')}")
        lines.append(f"- CTR: {_arrow(*get('ctr_pct'), 'pct')}")
        lines.append(f"- 비용(VAT 포함): {_arrow(*get('cost_vat'), 'money')}")
        lines.append(f"- 전환수: {_arrow(*get('conversions'), 'int')}")
        lines.append(f"- 전환매출액: {_arrow(*get('revenue'), 'money')}")
        lines.append(f"- ROAS: {_arrow(*get('roas_pct'), 'pct')}")
    else:
        m = er.metrics
        lines.append(f"이번 주 {ch}는 전주 데이터가 없어 단일 주 성과 기준으로 정리드립니다.")
        lines.append(f"- 노출수/클릭수/CTR: {format_int(m.get('impressions'))}회 / {format_int(m.get('clicks'))}회 / {format_float(m.get('ctr_pct'), 2)}%")
        lines.append(f"- 비용(VAT 포함): {format_int(m.get('cost_vat'))}원")
        lines.append(f"- 전환/매출/ROAS: {format_int(m.get('conversions'))}건 / {format_int(m.get('revenue'))}원 / {format_float(m.get('roas_pct'), 2)}%")

    if _needs_check(er.parse_warnings):
        lines.append("일부 지표 변동이 커 집계 기준(부가세 포함/제외, 전환 정의) 점검이 필요합니다.")

    lines.append("다음 주에는 비용 대비 전환/매출 기여 구간을 우선 점검·강화하겠습니다.")
    return lines[:9]

def render_report(
    enriched_records: List[EnrichedRecord],
    report_url: Optional[str],
    date_from: date,
    date_to: date
) -> Dict[str, str]:
    by_client: Dict[str, List[EnrichedRecord]] = {}
    for r in enriched_records:
        by_client.setdefault(r.client, []).append(r)

    outputs: Dict[str, str] = {}

    for client, recs in by_client.items():
        recs_sorted = sorted(
            recs,
            key=lambda x: CHANNEL_OUTPUT_ORDER.index(x.channel) if x.channel in CHANNEL_OUTPUT_ORDER else 999
        )

        lines: List[str] = []
        lines.append(f"{client} {date_from.isoformat()}~{date_to.isoformat()} 주간 보고서 전달드립니다.\n")
        lines.append("[보고서링크]")
        lines.append(report_url or "-")
        lines.append("(날짜는 선택하여 변경 가능합니다.)\n")

        for ch in CHANNEL_OUTPUT_ORDER:
            ch_recs = [r for r in recs_sorted if r.channel == ch]
            if not ch_recs:
                continue

            main = None
            for r in ch_recs:
                if r.item == "채널합":
                    main = r
                    break
            if main is None:
                main = ch_recs[0]

            m = main.metrics
            lines.append(f"■ {CHANNEL_DISPLAY.get(ch, ch)}")
            lines.append(f"노출수: {format_int(m.get('impressions'))}회")
            lines.append(f"클릭수: {format_int(m.get('clicks'))}회")
            lines.append(f"클릭률(CTR): {format_float(m.get('ctr_pct'), 2)}%")
            lines.append(f"총 광고비(VAT 포함): {format_int(m.get('cost_vat'))}원")
            lines.append(f"전환수: {format_int(m.get('conversions'))}건")
            lines.append(f"전환매출액: {format_int(m.get('revenue'))}원")
            lines.append(f"ROAS: {format_float(m.get('roas_pct'), 2)}%\n")

            lines.append("→ 코멘트")
            for c in _comment_lines(main):
                lines.append(c)
            lines.append("")

        outputs[client] = "\n".join(lines).rstrip() + "\n"

    return outputs


# ========================
# 8) 입력 메타/파이프라인
# ========================

def parse_input_meta(text: str) -> Tuple[Dict[str, str], str]:
    meta: Dict[str, str] = {}
    lines = text.splitlines()
    body_start = 0
    for i, ln in enumerate(lines):
        if BLOCK_START_RE.match(ln.strip()):
            body_start = i
            break
        if ":" in ln or "：" in ln:
            key, val = re.split(r"[:：]", ln, maxsplit=1)
            k = normalize_kv_key(key)
            meta[k] = val.strip()
    body = "\n".join(lines[body_start:]).lstrip("\n")
    return meta, body

def process_weekly_text(input_text: str, db_path: str = "ad_weekly_history.sqlite"):
    meta, body = parse_input_meta(input_text)

    report_url = meta.get("report_url") or meta.get("reporturl")
    week_start = meta.get("week_start") or meta.get("weekstart")
    date_from = meta.get("date_from") or meta.get("datefrom")
    date_to = meta.get("date_to") or meta.get("dateto")
    auto_mode = (meta.get("auto_mode") or meta.get("automode") or "lastmonday").strip().lower()
    if auto_mode not in ("monday", "lastmonday"):
        auto_mode = "lastmonday"

    ws, we = infer_week_range(
        week_start=week_start if week_start else None,
        date_from=date_from if date_from else None,
        date_to=date_to if date_to else None,
        auto_mode=auto_mode
    )

    blocks = split_blocks(body)
    all_records: List[Record] = []
    for b in blocks:
        all_records.extend(parse_block(b, ws, we))

    store = HistoryStore(db_path=db_path)
    enriched = build_wow(all_records, store)
    store.upsert_history(all_records)
    reports = render_report(enriched, report_url, ws, we)
    store.close()
    return reports, all_records, enriched


# ========================
# 9) 실행 엔트리
# ========================

def main():
    inp = Path("input.txt")
    if not inp.exists():
        print("input.txt 파일이 없습니다.")
        print("같은 폴더에 input.txt를 만들고 원문을 붙여넣어주세요.")
        raise SystemExit(1)

    text = inp.read_text(encoding="utf-8", errors="replace")
    reports, records, enriched = process_weekly_text(text, db_path="ad_weekly_history.sqlite")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    for client, txt in reports.items():
        (out_dir / f"{client}_report.txt").write_text(txt, encoding="utf-8")

    warn_lines: List[str] = []
    for r in enriched:
        if r.parse_warnings:
            warn_lines.append(f"[{r.client}][{r.channel}][{r.item}] {r.week_start.isoformat()}~{r.week_end.isoformat()}")
            for w in r.parse_warnings:
                warn_lines.append(f"- {w}")
            warn_lines.append("")
    (out_dir / "warnings.txt").write_text("\n".join(warn_lines).strip(), encoding="utf-8")

    print("완료: outputs 폴더에 보고서/경고 로그가 저장되었습니다.")
    print("누적 DB: ad_weekly_history.sqlite (같은 폴더에 유지하면 전주 비교가 이어집니다.)")

if __name__ == "__main__":
    main()
