import numpy as np
import heapq
import logging
from .guides import IBSNode
from .utils import evaluate_sequence

logger = logging.getLogger(__name__)


def reconstruct_solution_ibs(
        fixed_prefix_sequence,
        jobs_to_insert,
        fixed_suffix_sequence,
        processing_times,
        num_jobs,
        num_machines,
        guide_functions_handler,
        get_guide_function_for_recon_callback,
        ibs_beam_width_recon,
        max_discrepancies_recon,
        max_children_to_consider_for_lds_recon
):
    """
    Performs IBS reconstruction.
    get_guide_function_for_recon_callback: a function(alpha_for_walpha) that returns the guide function.
    """
    num_jobs_to_insert_total = len(jobs_to_insert)
    base_depth_for_recon = len(fixed_prefix_sequence)

    if num_jobs_to_insert_total == 0:
        full_sequence = list(fixed_prefix_sequence) + list(fixed_suffix_sequence)
        makespan, _ = evaluate_sequence(processing_times, full_sequence, num_machines)
        return full_sequence, makespan

    guide_functions_handler._clear_caches()

    _, initial_front_times = evaluate_sequence(processing_times, fixed_prefix_sequence, num_machines)

    alpha_for_guide_at_root = 0.0 / num_jobs_to_insert_total if num_jobs_to_insert_total > 0 else 1.0
    selected_guide_func_root = get_guide_function_for_recon_callback(alpha_for_walpha=alpha_for_guide_at_root)

    root_attrs = {
        "scheduled_sequence": list(fixed_prefix_sequence),
        "unscheduled_jobs": set(jobs_to_insert),
        "front_completion_times": initial_front_times.copy(),
        "depth": base_depth_for_recon
    }
    try:
        root_guide_val = selected_guide_func_root(root_attrs)
    except Exception as e:
        logger.error(f"Error calculating root guide value: {e}", exc_info=True)
        root_guide_val = float('inf')

    root_node = IBSNode(
        scheduled_sequence=list(fixed_prefix_sequence),
        unscheduled_jobs=set(jobs_to_insert),
        front_completion_times=initial_front_times.copy(),
        guide_value=root_guide_val,
        depth=base_depth_for_recon,
        discrepancies_used=0
    )

    current_beam_heap = [root_node]
    best_complete_insertion_sequence = None
    best_complete_insertion_makespan = float('inf')

    for i_job_insertion_step in range(num_jobs_to_insert_total):
        next_beam_candidates_heap = []
        if not current_beam_heap:
            logger.warning(
                f"IBS Recon: Beam died at insertion step {i_job_insertion_step + 1}/{num_jobs_to_insert_total}.")
            break

        num_parents_to_expand = len(current_beam_heap)

        recon_depth_step = i_job_insertion_step + 1
        alpha_for_this_level_guide = recon_depth_step / num_jobs_to_insert_total if num_jobs_to_insert_total > 0 else 1.0
        selected_guide_func = get_guide_function_for_recon_callback(alpha_for_walpha=alpha_for_this_level_guide)
        logger.debug(f"Recon step {i_job_insertion_step + 1}, alpha for walpha_guide: {alpha_for_this_level_guide:.3f}")

        for _ in range(num_parents_to_expand):
            if not current_beam_heap: break
            parent_node = heapq.heappop(current_beam_heap)

            children_with_scores = []
            for job_to_schedule_next in parent_node.unscheduled_jobs:
                new_scheduled_seq = parent_node.scheduled_sequence + [job_to_schedule_next]
                new_front_times_child = np.zeros(num_machines)
                new_front_times_child[0] = parent_node.front_completion_times[0] + processing_times[
                    job_to_schedule_next, 0]
                for m_idx in range(1, num_machines):
                    new_front_times_child[m_idx] = max(parent_node.front_completion_times[m_idx],
                                                       new_front_times_child[m_idx - 1]) + processing_times[
                                                       job_to_schedule_next, m_idx]

                child_attrs = {
                    "scheduled_sequence": new_scheduled_seq,
                    "unscheduled_jobs": parent_node.unscheduled_jobs - {job_to_schedule_next},
                    "front_completion_times": new_front_times_child,
                    "depth": len(new_scheduled_seq)
                }
                try:
                    child_guide_val = selected_guide_func(child_attrs)
                    children_with_scores.append({
                        "attrs": child_attrs, "guide_val": child_guide_val, "job_added": job_to_schedule_next
                    })
                except Exception as e:
                    logger.error(f"Error calculating child guide value: {e}", exc_info=True)
                    continue

            children_with_scores.sort(key=lambda x: x["guide_val"])

            for rank, child_info in enumerate(children_with_scores):
                if rank >= max_children_to_consider_for_lds_recon: break

                discrepancies_for_child = parent_node.discrepancies_used + rank

                if discrepancies_for_child <= max_discrepancies_recon:
                    child_node = IBSNode(
                        scheduled_sequence=child_info["attrs"]["scheduled_sequence"],
                        unscheduled_jobs=child_info["attrs"]["unscheduled_jobs"],
                        front_completion_times=child_info["attrs"]["front_completion_times"],
                        guide_value=child_info["guide_val"],
                        depth=child_info["attrs"]["depth"],
                        discrepancies_used=discrepancies_for_child
                    )
                    heapq.heappush(next_beam_candidates_heap, child_node)
                    # logger.debug(f"  Added child (Job {child_info['job_added']}, Guide: {child_info['guide_val']:.2f}, Disc: {discrepancies_for_child}) from parent (Depth: {parent_node.depth}, PDisc: {parent_node.discrepancies_used})")

        current_beam_heap = []
        for _ in range(ibs_beam_width_recon):
            if not next_beam_candidates_heap: break
            heapq.heappush(current_beam_heap, heapq.heappop(next_beam_candidates_heap))

        # logger.debug(f"IBS Recon Step {i_job_insertion_step+1}: Beam size {len(current_beam_heap)}.")
        # if current_beam_heap:
        #      logger.debug(f"Best in beam: (Guide: {current_beam_heap[0].guide_value:.2f}, Disc: {current_beam_heap[0].discrepancies_used})")

    if not current_beam_heap:
        logger.warning("IBS Recon: Beam empty after all insertion steps. No solution found by IBS.")
        return None, float('inf')

    for final_insertion_node in current_beam_heap:
        if len(final_insertion_node.scheduled_sequence) != (base_depth_for_recon + num_jobs_to_insert_total):
            logger.error(f"IBS Recon: Mismatch in final node length.")
            continue

        reconstructed_middle_part = final_insertion_node.scheduled_sequence[base_depth_for_recon:]
        full_reconstructed_sequence = list(fixed_prefix_sequence) + reconstructed_middle_part + list(
            fixed_suffix_sequence)

        if len(full_reconstructed_sequence) != num_jobs:
            logger.error(f"IBS Recon: Final constructed sequence length error.")
            continue

        makespan, _ = evaluate_sequence(processing_times, full_reconstructed_sequence, num_machines)

        if makespan < best_complete_insertion_makespan:
            best_complete_insertion_makespan = makespan
            best_complete_insertion_sequence = full_reconstructed_sequence

    if best_complete_insertion_sequence:
        logger.debug(f"IBS Recon successful. Best M: {best_complete_insertion_makespan:.2f}")
    else:
        logger.warning("IBS Recon: No valid complete sequence found from final beam.")
        return None, float('inf')

    return best_complete_insertion_sequence, best_complete_insertion_makespan