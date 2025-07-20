import numpy as np
from .utils import evaluate_sequence  # Use relative import
import logging


def neh_heuristic(processing_times, num_jobs, num_machines):
    """
    NEH heuristic for the flow shop scheduling problem.
    Returns the sequence and its makespan.
    """
    if num_jobs == 0:
        return [], 0

    job_indices = list(range(num_jobs))
    job_sum_pts = {job_idx: np.sum(processing_times[job_idx, :]) for job_idx in job_indices}
    sorted_jobs = sorted(job_indices, key=lambda x: job_sum_pts[x], reverse=True)

    current_sequence = [sorted_jobs[0]]
    if num_jobs == 1:
        makespan, _ = evaluate_sequence(processing_times, current_sequence, num_machines)
        return current_sequence, makespan

    # Add the second job
    job2 = sorted_jobs[1]
    seq1 = [current_sequence[0], job2]
    m1,_   = evaluate_sequence(processing_times, seq1, num_machines)
    seq2 = [job2, current_sequence[0]]
    m2,_   = evaluate_sequence(processing_times, seq2, num_machines)

    if m1 <= m2:
        current_sequence = seq1
        current_makespan = m1
    else:
        current_sequence = seq2
        current_makespan = m2

    # Add remaining jobs
    for i in range(2, num_jobs):
        job_to_insert = sorted_jobs[i]
        best_makespan_for_insertion = float('inf')
        best_sequence_for_insertion = []

        for k in range(len(current_sequence) + 1):  # Try all k+1 positions
            temp_sequence = current_sequence[:k] + [job_to_insert] + current_sequence[k:]
            makespan,_ = evaluate_sequence(processing_times, temp_sequence, num_machines)

            if makespan < best_makespan_for_insertion:
                best_makespan_for_insertion = makespan
                best_sequence_for_insertion = temp_sequence
            # Tie-breaking: if makespans are equal, prefer earlier insertion (already handled by <)
            # Or, if makespans are equal and position is the same, the first one found is kept.

        current_sequence = best_sequence_for_insertion
        current_makespan = best_makespan_for_insertion

    logging.debug(f"NEH final sequence: {current_sequence}, makespan: {current_makespan}")
    return current_sequence, current_makespan


def local_search_insertion(processing_times, initial_sequence, num_jobs, num_machines):
    """
    Performs a best insertion local search (RI - Reinsertion) on the initial_sequence.
    Returns the improved sequence and its makespan.
    """
    if num_jobs <= 1:  # Nothing to reinsert
        m, _ = evaluate_sequence(processing_times, initial_sequence, num_machines)
        return list(initial_sequence), m

    current_sequence = list(initial_sequence)
    current_makespan, _ = evaluate_sequence(processing_times, current_sequence, num_machines)

    improved = True
    iteration_count = 0
    max_iterations = num_jobs * num_jobs  # Heuristic limit to prevent excessive run time for large N

    while improved and iteration_count < max_iterations:
        improved = False
        iteration_count += 1

        best_sequence_in_iteration = list(current_sequence)
        best_makespan_in_iteration = current_makespan

        for i in range(num_jobs):  # For each job to be potentially re-inserted
            job_to_reinsert = current_sequence[i]

            temp_sequence_without_job = current_sequence[:i] + current_sequence[i + 1:]

            # Try inserting job_to_reinsert at all possible positions (j)
            # including its original effective position
            for j in range(num_jobs):  # N positions in a sequence of N jobs (0 to N-1)
                # If j == i, it means inserting back into the same slot (relative to N-1 other jobs)

                new_test_sequence = temp_sequence_without_job[:j] + [job_to_reinsert] + temp_sequence_without_job[j:]

                if not new_test_sequence: continue  # Should not happen if num_jobs > 0

                test_makespan, _ = evaluate_sequence(processing_times, new_test_sequence, num_machines)

                if test_makespan < best_makespan_in_iteration:
                    best_makespan_in_iteration = test_makespan
                    best_sequence_in_iteration = list(new_test_sequence)
                    improved = True  # Found an improvement in this pass over job i

        if improved:
            current_sequence = list(best_sequence_in_iteration)
            current_makespan = best_makespan_in_iteration
            logging.debug(f"LS Insertion: Improved to {current_makespan:.2f} in iter {iteration_count}")

    logging.info(
        f"Local Search (Insertion) finished. Initial M: (pass if not available), Final M: {current_makespan:.2f} after {iteration_count} main loops.")
    return current_sequence, current_makespan