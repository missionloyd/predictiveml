from joblib import Parallel, delayed

def print_progress(title, batch_number, total_batches, progress):
    filled_length = int(progress // 10)
    bar = '#' * filled_length + ' ' * (10 - filled_length)

    batch_format = f"Batch {batch_number}/{total_batches}"
    progress_format = f"[{progress:<4.1f}%]"
    
    print(f"{batch_format: <14s}{progress_format: <10s}|{bar}| ({title})")



def process_batch_args(title, arguments, batch_size, func):
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
        print_progress(title, batch_number + 1, total_batches, progress_before)

        batch_results = Parallel(n_jobs=-1, prefer="processes")(delayed(func)(arg) for arg in batch_arguments)
        results.extend(batch_results)

        batch_number += 1

    progress_after = (batch_number / total_batches) * 100
    print_progress(title, batch_number, total_batches, progress_after)

    return results




# def process_batch_args(title, arguments, batch_size, func):
#     results = []
#     for batch_start in range(0, len(arguments), batch_size):
#         batch_end = batch_start + batch_size
#         batch_arguments = arguments[batch_start:batch_end]
#         batch_results = Parallel(n_jobs=-1, prefer="processes")(delayed(func)(arg) for arg in batch_arguments)
#         results.extend(batch_results)
#     return results
