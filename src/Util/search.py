import concurrent
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import glob



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

