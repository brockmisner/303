# execution_safety.py — Last-moment edge integrity guard
"""
Protects against:
- Microstructure edge collapse between signal and submission
- Late-window spread explosions
- Signal-book mismatch

Usage:
    ok, edge = validate_execution_edge("BUY", 0.45, 0.52, fee_per_share, 0.01)
    if not ok:
        logger.info("SKIP_EDGE_ERODED")
"""


def validate_execution_edge(
    side: str,
    exec_price: float,
    model_prob: float,
    fee_fn,
    min_edge: float,
) -> tuple:
    """
    Recompute edge just before submission.
    Blocks trade if edge < min_edge.

    Args:
        side: "BUY" or "SELL"
        exec_price: actual limit price being submitted
        model_prob: model probability for THIS token
        fee_fn: callable(price) -> fee
        min_edge: minimum acceptable edge

    Returns:
        (ok: bool, edge: float)
    """
    fee = fee_fn(exec_price)

    if side == "BUY":
        edge = model_prob - exec_price - fee
    else:
        edge = exec_price - model_prob - fee

    return edge >= min_edge, round(edge, 6)


def book_health_score(snap_age_ms: float, stale_ms: float = 800.0) -> float:
    """
    Book health score: 1.0 = fresh, 0.0 = stale.
    Use to scale Kelly when book quality degrades.
    """
    if snap_age_ms <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (snap_age_ms / stale_ms)))
