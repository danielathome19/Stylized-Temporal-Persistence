"""Exact requested RD payload for the implemented pixel-event GPU layout."""


def allocation(width, height, capacity):
    assert width > 0 and height > 0 and capacity > 0
    p = width * height
    return {
        "previous_rgba32f": 16 * p,
        "signed_event_pool": 16 * capacity,
        "pixel_birth_metadata": 8 * capacity,
        "selection_heap": 4 * capacity,
        "event_links": 4 * capacity,
        "pixel_heads": 4 * p,
        "controls": 16,
        "event_priorities": 4 * capacity,
        "incoming_priorities": 4 * p,
    }


def capacity_for_budget(width, height, budget):
    return max(0, (budget - 24 * width * height - 16) // 36)
