import concurrent
import glob
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor


def order_path_list(group):
    """Return a list indexed by plot_id where present, preserving others at the end.

    - Safely handles gaps and large plot IDs by using a dict first.
    - Keeps non-matching paths appended after indexed ones.
    """
    indexed: dict[int, str] = {}
    rest: list[str] = []
    for path in group:
        match = re.search(r"plot_(\d+)", path)
        if match:
            try:
                plot_id = int(match.group(1))
                indexed[plot_id] = path
            except ValueError:
                rest.append(path)
        else:
            rest.append(path)

    # Rebuild ordered list with gaps filled by None
    ordered: list[str | None] = []
    if indexed:
        for i in range(max(indexed.keys()) + 1):
            ordered.append(indexed.get(i))
    # Append leftovers
    ordered.extend(rest)
    return ordered


IGNORE_DIRS = {"System Volume Information"}
PATTERN_TMPL = "*{obj}*.parquet"


def search_directory(directory, objective):
    """
    Searches a single directory for files matching the objective.
    """
    try:
        # Use glob to find matching .parquet files in the current directory
        pattern = os.path.join(directory, f"*{objective}*.parquet")
        matches = glob.glob(pattern)
        if matches:
            return matches
    except OSError as e:
        # Handle potential permission errors when accessing directories
        # print(f"Could not access {directory}: {e}")
        pass
    return None


def optimized_recursive_search(folders, objective, start_dir, remove_unkwown=True):
    """
    Search for parquet files matching an objective in relevant directories.

    Args:
        folders (list): List of folder name fragments to identify relevant directories
        objective (str): String that should be in the filename
        start_dir (str): Starting directory for the search

    Returns:
        dict: Dictionary with week IDs as keys and lists of file paths as values
    """

    logging.info(f"Starting search from {start_dir}")
    logging.info(f"Looking for files containing '{objective}' in folders related to {folders}")

    results_by_week = {}
    stats = {"directories_checked": 0, "files_found": 0, "weeks_found": set()}

    # Walk through all directories
    for root, dirs, _ in os.walk(start_dir):
        stats["directories_checked"] += 1

        # Skip system folders and hidden directories
        dirs[:] = [d for d in dirs if "$" not in d and d not in IGNORE_DIRS]

        is_relevant = any(folder in root for folder in folders) if folders else True

        week_match = re.search(r"week\d+", root)
        week_id = week_match.group() if week_match else None

        if is_relevant or week_id:
            logging.debug(f"Checking directory: {root}")

            pattern = os.path.join(root, f"*{objective}*.parquet")
            matching_files = glob.glob(pattern)

            if matching_files:
                if not week_id:
                    for file in matching_files:
                        file_week_match = re.search(r"week\d+", file)
                        if file_week_match:
                            week_id = file_week_match.group()
                            break

                if not week_id:
                    week_id = "unknown"

                if week_id not in results_by_week:
                    results_by_week[week_id] = []
                    stats["weeks_found"].add(week_id)

                results_by_week[week_id].extend(matching_files)
                stats["files_found"] += len(matching_files)

                logging.info(f"Found {len(matching_files)} files for {week_id} in {root}")

    if remove_unkwown:
        results_by_week.pop("unknown", None)

    logging.info(f"Search complete: checked {stats['directories_checked']} directories")
    logging.info(f"Found {stats['files_found']} files across {len(stats['weeks_found'])} weeks")
    logging.info(f"Weeks found: {sorted(list(stats['weeks_found']))}")

    return results_by_week
