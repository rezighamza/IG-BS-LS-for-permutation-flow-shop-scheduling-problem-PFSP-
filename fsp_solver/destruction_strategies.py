import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _relatedness_sum_abs_proc_diff(job1_idx, job2_idx, processing_times, num_jobs):
    """ Calculates relatedness based on sum of absolute differences in processing times. """
    if job1_idx >= num_jobs or job2_idx >= num_jobs:
        logger.error(f"Job index out of bounds in relatedness: {job1_idx}, {job2_idx}")
        return float('inf')

    diff = np.sum(np.abs(processing_times[job1_idx, :] - processing_times[job2_idx, :]))
    return diff


def destroy_solution_shaw(current_sequence_list, num_to_destroy, processing_times, num_jobs):
    """
    Performs Shaw-like removal.
    Returns: (fixed_part_remaining_ordered, list_of_removed_jobs, empty_suffix_list)
    """
    if not current_sequence_list or num_to_destroy == 0 or len(current_sequence_list) <= num_to_destroy:
        logger.debug("Shaw Destroy: Not enough jobs to destroy or num_to_destroy is 0.")
        return list(current_sequence_list), [], []

    seed_job_pos = random.randrange(len(current_sequence_list))
    seed_job_actual_idx = current_sequence_list[seed_job_pos]

    removed_jobs_set = {seed_job_actual_idx}

    relatedness_scores = []
    for job_idx_in_seq in current_sequence_list:
        if job_idx_in_seq == seed_job_actual_idx:
            continue
        rel_val = _relatedness_sum_abs_proc_diff(seed_job_actual_idx, job_idx_in_seq, processing_times, num_jobs)
        relatedness_scores.append((rel_val, job_idx_in_seq))

    relatedness_scores.sort(key=lambda x: x[0])

    for i in range(min(num_to_destroy - 1, len(relatedness_scores))):
        removed_jobs_set.add(relatedness_scores[i][1])

    fixed_part_remaining_ordered = [job for job in current_sequence_list if job not in removed_jobs_set]

    logger.debug(
        f"Shaw Destroy: Removed {len(removed_jobs_set)} jobs. Seed: {seed_job_actual_idx}. Removed: {removed_jobs_set}")
    return fixed_part_remaining_ordered, list(removed_jobs_set), []


def destroy_solution_block(current_sequence_list, num_to_destroy):
    """
    Performs block removal.
    Returns: (prefix, destroyed_jobs_list, suffix)
    """
    num_jobs_total = len(current_sequence_list)
    # effective_k_destroy is already calculated before calling this in solver_core
    # but good to have safety here if called directly
    if num_jobs_total <= 1 or num_to_destroy == 0 or num_jobs_total <= num_to_destroy:
        logger.debug(
            f"Block Destroy: Seq len {num_jobs_total}, num_to_destroy {num_to_destroy}. No effective destruction.")
        return list(current_sequence_list), [], []

    start_index_max = num_jobs_total - num_to_destroy
    start_index_for_removal = random.randint(0, start_index_max)

    prefix = current_sequence_list[:start_index_for_removal]
    destroyed_jobs = current_sequence_list[start_index_for_removal: start_index_for_removal + num_to_destroy]
    suffix = current_sequence_list[start_index_for_removal + num_to_destroy:]

    logger.debug(f"Block Destroyed {len(destroyed_jobs)} jobs from index {start_index_for_removal}.")
    return prefix, destroyed_jobs, suffix