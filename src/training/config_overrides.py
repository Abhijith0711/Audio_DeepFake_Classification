"""Apply dotted ``key=value`` overrides on top of a loaded YAML config."""


def parse_override_value(raw):
    text = str(raw).strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if any(token in text for token in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def apply_overrides(config, overrides):
    """Set ``a.b=value`` paths. Values are parsed as bool, null, float, or int."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError(f"--set expects key=value, got {item!r}")
        node = config
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = parse_override_value(raw)
    return config
