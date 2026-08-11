from main.services.plate_enhance import (
    expand_bbox,
    normalize_plate_text,
    PlateVoteTracker,
    iou,
)


def test_normalize_mercosur():
    assert normalize_plate_text("ab123cd") == "AB123CD"
    assert normalize_plate_text("AB12OCD") == "AB120CD"  # O→0 en dígitos


def test_normalize_old_format():
    assert normalize_plate_text("abc123") == "ABC123"
    assert normalize_plate_text("ABCI23") == "ABC123"  # I→1 en dígitos


def test_expand_bbox():
    x1, y1, x2, y2 = expand_bbox(100, 100, 200, 140, (480, 640, 3), pad_ratio=0.1)
    assert x1 < 100 and y1 < 100 and x2 > 200 and y2 > 140


def test_vote_tracker_majority():
    tracker = PlateVoteTracker(iou_threshold=0.5, maxlen=10)
    box = (10, 10, 100, 40)
    tracker.add(box, "AB123CD", 0.5)
    tracker.add(box, "AB123CD", 0.6)
    tracker.add((12, 12, 98, 38), "AB129CD", 0.4)
    best = tracker.add(box, "AB123CD", 0.7)
    assert best == "AB123CD"


def test_iou_overlap():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
