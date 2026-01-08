"""Argument parsing utilities."""


def parse_ball_tiles(value: str) -> tuple[int, int] | None:
    """Parse ball tile grid argument (e.g., '2x2').

    Args:
        value: Tile grid string like '2x2' or 'none'

    Returns:
        Tuple of (rows, cols) or None

    Raises:
        ValueError: If format is invalid
    """
    if not value:
        return None
    text = value.strip().lower()
    if text in {"none", "off", "0"}:
        return None
    if "x" not in text:
        raise ValueError("ball-tiles must be like 2x2")
    parts = text.split("x")
    if len(parts) != 2:
        raise ValueError("ball-tiles must be like 2x2")
    rows = int(parts[0])
    cols = int(parts[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("ball-tiles must be like 2x2")
    return rows, cols
