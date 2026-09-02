"""Grid ground-truth constants and homography behaviour.

The numeric ground truth (18x32, river rows 15-16, row-0 restriction, mirror
31-y) is asserted here. If empirical calibration ever disagrees, the
calibration wins and the discrepancy must be reported, not reconciled.
"""
import numpy as np

from cr_perception import geometry as g


def test_grid_constants():
    assert g.COLS == 18 and g.ROWS == 32
    assert list(g.OWN_ROWS) == list(range(0, 15))
    assert g.RIVER_ROWS == (15, 16)
    assert list(g.ENEMY_ROWS) == list(range(17, 32))
    assert list(g.ROW0_COLS) == [6, 7, 8, 9, 10, 11]
    assert g.mirror_row(0) == 31 and g.mirror_row(14) == 17 and g.mirror_row(15) == 16


def _trapezoid():
    # own side (bottom) wider than enemy side (top), vertically compressed: a
    # perspective-like arena of 18x32 tiles drawn 600px wide at the bottom.
    return [(50, 900), (650, 900), (600, 200), (100, 200)]  # bl, br, tr, tl


def test_homography_round_trip_and_perspective():
    h = g.Homography.from_corners(_trapezoid())
    for c in (0, 5, 17):
        for r in (0, 15, 31):
            px, py = h.tile_to_pixel(c, r)
            assert h.pixel_to_tile(px, py) == (c, r)
    # rows are NOT equally tall on screen: bottom row taller than top row
    _, y0 = h.tile_to_pixel(9, 0, centre=False)
    _, y1 = h.tile_to_pixel(9, 1, centre=False)
    _, y30 = h.tile_to_pixel(9, 30, centre=False)
    _, y31 = h.tile_to_pixel(9, 31, centre=False)
    assert (y0 - y1) > (y30 - y31)
    # a linear scale would put row 16 at the pixel midpoint; the homography must not
    _, ymid = h.tile_to_pixel(9, 16, centre=False)
    assert abs(ymid - (900 + 200) / 2) > 1.0


def test_bottom_centre_rule():
    h = g.Homography.from_corners(_trapezoid())
    # a tall sprite standing on tile (9, 5): its bbox bottom is on that tile,
    # its centre is several rows further up the screen (toward the enemy).
    fx, fy = h.tile_to_pixel(9, 5)
    bbox = (fx - 20, fy - 120, fx + 20, fy)
    assert g.bbox_to_tile(h, bbox) == (9, 5)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    assert h.pixel_to_tile(cx, cy)[1] > 5  # centre rule would bias toward the enemy side


def test_legal_mask_static_and_after_tower_destroyed():
    m = g.legal_placement_mask(g.TowerState())
    assert m.shape == (18, 32)
    assert m[:, 1:15].all()
    assert not m[:, 15:].any()
    assert m[6:12, 0].all() and not m[0:6, 0].any() and not m[12:, 0].any()
    m2 = g.legal_placement_mask(g.TowerState(enemy_left=False))
    assert m2[0:9, 17:21].all() and not m2[9:, 17:21].any()
    assert m2[3, 15] and m2[3, 16] and not m2[14, 15]
    assert g.mask_hash(m) != g.mask_hash(m2)


def test_from_correspondences_matches_from_corners():
    corners = _trapezoid()
    h = g.Homography.from_corners(corners)
    px = [h.tile_to_pixel(c, r, centre=False) for c, r in [(9, 0.5), (3.5, 4.3), (14.5, 4.3), (9, 26.3), (3.5, 23.3), (14.5, 23.3)]]
    tiles = [(9, 0.5), (3.5, 4.3), (14.5, 4.3), (9, 26.3), (3.5, 23.3), (14.5, 23.3)]
    h2 = g.Homography.from_correspondences(px, tiles)
    for c, r in [(0, 0), (17, 31), (8, 16)]:
        a = h.tile_to_pixel(c, r)
        b = h2.tile_to_pixel(c, r)
        assert np.allclose(a, b, atol=0.5)
