import concurrent
import logging
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor

IGNORE_DIRS = {"System Volume Information"}
PATTERN_TMPL = "*{obj}*.parquet"
import os
import glob
from pathlib import Path



def logging_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("process.log", encoding='utf-8'),  # Note the encoding parameter
            logging.StreamHandler()
        ]
    )

def recursive_search(folders, objective,start= r'D:\**' , result = None):
    if result is None:
        result = []

    if start.count(r'D:\**') != 0:
        all_paths = glob.glob(r'D:\**')
    else:
        all_paths = glob.glob(os.path.join(start, '*'))

    directories = [path for path in all_paths
                   if os.path.isdir(path) \
                   and path.count('$') == 0 \
                   and path.count('System Volume Information') == 0 \
                   or any(sub in path for sub in folders)
                   ]

    for directory in directories:
        dfs = glob.glob(os.path.join(directory,'*'))
        matches = [f for f in dfs if objective in f and f.endswith('.parquet')]
        if matches:
            result.append(matches)
        recursive_search(folders=folders, objective = objective, start = directory, result = result)
    return result


def find_parquets(base_dir, objective):
    results = []
    for p in Path(base_dir).rglob('*.parquet'):
        if objective in p.name:
            results.append(str(p))
    return results

def search_directory(directory, objective):
    """
    Searches a single directory for files matching the objective.
    """
    try:
        # Use glob to find matching .parquet files in the current directory
        pattern = os.path.join(directory, f'*{objective}*.parquet')
        matches = glob.glob(pattern)
        if matches:
            return matches
    except OSError as e:
        # Handle potential permission errors when accessing directories
        # print(f"Could not access {directory}: {e}")
        pass
    return None



def _scan_one(root: Path, obj: str) -> list[str]:
    pat = PATTERN_TMPL.format(obj=obj)
    # rglob streams entries via OS APIs (≈os.scandir); no Python recursion cost
    return [str(p) for p in root.rglob(pat)
            if "$" not in p.parts and not (set(p.parts) & IGNORE_DIRS)]

def fast_scan(base="D:\\", objective="plot_", workers=8) -> list[str]:
    roots = [p for p in Path(base).iterdir() if p.is_dir()]
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(_scan_one, r, objective): r for r in roots}
        return list(chain.from_iterable(f.result() for f in as_completed(futures)))

def optimized_recursive_search(folders, objective, start=r'D:\\'):
    """
    Recursively searches for parquet files containing a specific objective string
    in their names within specified folders, using multithreading.

    Args:
        folders (list): A list of substrings. A directory path must contain one of these
                        substrings to be included in the search.
        objective (str): The string that the target .parquet filenames must contain.
        start (str): The starting directory for the search.

    Returns:
        list: A list of lists, where each inner list contains matching file paths
              from a single directory.
    """
    results = []
    # Use os.walk for efficient, top-down directory traversal.
    # It avoids deep recursion and is generally faster than recursive glob.
    directories_to_search = []
    for root, dirs, _ in os.walk(start):
        # Exclude special directories from the search to avoid errors and cycles.
        dirs[:] = [d for d in dirs if '$' not in d and 'System Volume Information' not in d]

        # Check if the current directory path matches the criteria.
        if any(sub in root for sub in folders):
            directories_to_search.append(root)

    # Use a ThreadPoolExecutor to search directories in parallel.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a future for each directory search task.
        future_to_dir = {executor.submit(search_directory, directory, objective): directory for directory in directories_to_search}

        for future in concurrent.futures.as_completed(future_to_dir):
            try:
                matches = future.result()
                if matches:
                    results.append(matches)
            except Exception as e:
                # Handle exceptions that might occur within a thread.
                print(f"An error occurred during search: {e}")

    return results

def main():


    print(recursive_search(folders = ['metashape','products_uav_data','output','extract','polygon_df'], objective="plot_"))

    print(len(find_parquets(r'D:\\', objective="plot_")))

if __name__ == "__main__":
    main()
