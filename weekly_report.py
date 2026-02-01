# -*- coding: utf-8 -*-
"""
주간 광고 보고서 자동 생성기 (EXE 빌드용)
- input.txt에 주간 원문을 붙여넣고 exe(또는 python) 실행하면 outputs 폴더에 업체별 보고서 txt 생성
- 누적 HISTORY: ad_weekly_history.sqlite (동일 폴더에 유지하면 WoW 비교 가능)
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

# 헤더 표준화 매핑
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

# A 포맷 키:값 유사어
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

def infer_week_range(week_start: Optional[str] = None,
                     date_from: Optional[str] = None,
                     date_to: Optional[str] = None,
                     auto_mode: str = "monday") -> Tuple[date, date]:
    """
    - week_start 있으면 week_start~week_start+6
    - date_from/date_to 있으면 그대로
    - 아무것도 없으면:
      auto_mode="monday" => 이번 주(월~일)
      auto_mode="lastmonday" => 지난 주(월~일)
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

def parse_number(token: str) -> Optional[float]:
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
            blocks.append(Block(cur_client.strip(), cur_channel.strip(), body_text))
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
    std_fields = []
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

