from moscato.object_crops import choose_crop


def obj(box):
    return {'class': 'targetobject', 'box_xyxy': box}


def hand(offset=(0, 0, 0), contact=3):
    return {'class': 'hand', 'box_xyxy': [0, 0, 10, 10],
            'contact_state_index': contact, 'offset': offset}


def test_no_contact_means_missing_not_full_frame():
    assert choose_crop([obj([0, 0, 20, 20]), hand(contact=0)], 100, 100) is None
    assert choose_crop([hand()], 100, 100) is None


def test_offset_selects_object_not_nearest_to_hand():
    # Endpoint x=85, rather than hand center x=5.
    ds = [obj([0, 0, 10, 10]), obj([80, 0, 90, 10]), hand((.08, .1, 0))]
    assert choose_crop(ds, 100, 100) == [80, 0, 91, 11]


def test_union_and_image_bounds():
    ds = [obj([-1, -2, 10, 10]), obj([80, 0, 110, 20]), hand(), hand((.09, .1, 0))]
    assert choose_crop(ds, 100, 100) == [0, 0, 100, 21]
