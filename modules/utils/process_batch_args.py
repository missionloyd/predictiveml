from joblib import Parallel, delayed
from modules.logging_methods.main import logger

def print_progress(title, batch_number, total_batches, progress):
    filled_length = int(progress // 10)
    bar = '#' * filled_length + ' ' * (10 - filled_length)

    batch_format = f"Batch {batch_number}/{total_batches}"
    progress_format = f"[{progress:<4.1f}%]"

    output = f"{batch_format: <14s}{progress_format: <10s}|{bar}| ({title})"
    logger(output)
    return output

def process_batch_args(title, arguments, func, batch_size, n_jobs):
    logger('')
    results = []
    if not arguments:  # Check if the arguments list is empty
        return results

    total_batches = (len(arguments) + batch_size - 1) // batch_size  # Calculate the total number of batches
    if total_batches == 0:  # Check if total_batches is zero
        return results

    batch_number = 0

    for batch_start in range(0, len(arguments), batch_size):
        batch_end = min(batch_start + batch_size, len(arguments))
        batch_arguments = arguments[batch_start:batch_end]

        progress_before = (batch_number / total_batches) * 100
        print_progress(title, batch_number, total_batches, progress_before)

        # Acquire the lock to ensure exclusive access to Parallel execution
        batch_results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(func)(arg) for arg in batch_arguments)

        results.extend(batch_results)

        batch_number += 1

    progress_after = (batch_number / total_batches) * 100
    print_progress(title, batch_number, total_batches, progress_after)
    logger('')
    return results
