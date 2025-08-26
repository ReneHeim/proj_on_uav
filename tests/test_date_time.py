from datetime import datetime

from src.Utils.extract_data.date_time import convert_to_timezone


def test_convert_to_timezone_offsets():
    dt = datetime(2024, 1, 1, 12, 0, 0)
    z = convert_to_timezone(dt, "UTC")
    assert z.tzinfo is not None
    z2 = convert_to_timezone(dt, "UTC+1")
    assert z2.utcoffset().total_seconds() == 3600
    z3 = convert_to_timezone(dt, "UTC-2")
    assert z3.utcoffset().total_seconds() == -7200


def test_convert_to_timezone_named():
    dt = datetime(2024, 1, 1, 12, 0, 0)
    z = convert_to_timezone(dt, "Europe/Berlin")
    assert z.tzinfo is not None
