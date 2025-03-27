import logging
import pytz


def convert_to_timezone(dt, timezone_str):
    """
    Convert a datetime to the specified timezone.

    Args:
        dt (datetime): Naive datetime object to localize
        timezone_str (str): Timezone string in format like "Europe/Berlin", "CET", or "UTC+1"

    Returns:
        datetime: Timezone-aware datetime object
    """
    try:
        # Handle UTC with offset format (e.g., "UTC+1", "UTC-5")
        if timezone_str.startswith("UTC") and ('+' in timezone_str or '-' in timezone_str):
            if '+' in timezone_str:
                offset = int(timezone_str.split('+')[1])
                tz = pytz.FixedOffset(offset * 60)  # Convert hours to minutes
            elif '-' in timezone_str:
                offset = int(timezone_str.split('-')[1])
                tz = pytz.FixedOffset(-offset * 60)  # Negative offset
            return dt.replace(tzinfo=tz)
        # Handle plain "UTC"
        elif timezone_str == "UTC":
            return dt.replace(tzinfo=pytz.UTC)
        # Handle standard timezone names
        else:
            tz = pytz.timezone(timezone_str)
            return tz.localize(dt)
    except Exception as e:
        logging.warning(f"Error setting timezone '{timezone_str}': {e}. Falling back to UTC.")
        return dt.replace(tzinfo=pytz.UTC)