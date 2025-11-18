def safe_truncate(s, max_len=1000):
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= max_len else s[:max_len-3] + '...'
