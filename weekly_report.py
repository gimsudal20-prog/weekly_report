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

    # 1) 값 개수 정상
    if len(value_tokens) == len(std_fields):
        for f, tok in zip(std_fields, value_tokens):
            if f:
                metrics[f] = parse_number(tok)
        return item_label, metrics, warnings

    # 2) 값 1개 부족 & daily_budget 존재: daily_budget NULL 보정
    if len(value_tokens) == len(std_fields) - 1 and "daily_budget" in std_fields:
        idx = std_fields.index("daily_budget")
        vt = value_tokens[:idx] + [None] + value_tokens[idx:]
        for f, tok in zip(std_fields, vt):
            if f:
                metrics[f] = parse_number(tok) if tok is not None else None
        warnings.append("daily_budget 값 누락으로 NULL 처리")
        return item_label, metrics, warnings

    # 3) 그 외: 가능한 만큼만 앞에서 매핑
    warnings.append(f"헤더({len(std_fields)})-값({len(value_tokens)}) 개수 불일치")
    for f, tok in zip(std_fields, value_tokens):
        if f:
            metrics[f] = parse_number(tok)
    return item_label, metrics, warnings
