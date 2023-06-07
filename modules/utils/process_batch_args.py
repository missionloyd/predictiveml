from joblib import Parallel, delayed

def process_batch_args(arguments, batch_size, func):
    results = []
    for batch_start in range(0, len(arguments), batch_size):
        batch_end = batch_start + batch_size
        batch_arguments = arguments[batch_start:batch_end]
        batch_results = Parallel(n_jobs=-1, prefer="processes")(delayed(func)(arg) for arg in batch_arguments)
        results.extend(batch_results)
    return results