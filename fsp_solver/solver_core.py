import numpy as np
import time
import logging
import random  # Ensure random is imported if not already for the new logic

# Relative imports from the same package
from .utils import evaluate_sequence
from .heuristics import neh_heuristic, local_search_insertion
from .guides import GuideFunctions
from .destruction_strategies import destroy_solution_block, destroy_solution_shaw
from .reconstruction_ibs import reconstruct_solution_ibs

DEFAULT_STAGNATION_LIMIT = 50
DEFAULT_PERTURB_K_DESTROY_PERCENT = 0.6
DEFAULT_IG_ITERATIONS = 100
DEFAULT_K_DESTROY_PERCENT = 0.20
DEFAULT_IBS_BEAM_WIDTH_RECON = 5
DEFAULT_GLOBAL_TIME_LIMIT_SECONDS = 60
DEFAULT_MAX_DISCREPANCIES_RECON = 1
DEFAULT_MAX_CHILDREN_LDS_RECON = 3
DEFAULT_DESTRUCTION_METHOD = 'block'
DEFAULT_APPLY_LOCAL_SEARCH_ON_BEST = True
# --- New defaults for boundary job addition ---
DEFAULT_ADD_BOUNDARY_JOBS = False  # Disabled by default
DEFAULT_NUM_BOUNDARY_JOBS_TO_ADD = 1


class IG_IBS_Solver:
    def __init__(self, processing_times,
                 ig_iterations=DEFAULT_IG_ITERATIONS,
                 k_destroy_percent=DEFAULT_K_DESTROY_PERCENT,
                 ibs_beam_width_recon=DEFAULT_IBS_BEAM_WIDTH_RECON,
                 guide_choice_recon='walpha_fwd',
                 global_time_limit=DEFAULT_GLOBAL_TIME_LIMIT_SECONDS,
                 max_discrepancies_recon=DEFAULT_MAX_DISCREPANCIES_RECON,
                 max_children_lds_recon=DEFAULT_MAX_CHILDREN_LDS_RECON,
                 destruction_method=DEFAULT_DESTRUCTION_METHOD,
                 apply_local_search_on_best=DEFAULT_APPLY_LOCAL_SEARCH_ON_BEST,
                 stagnation_limit=DEFAULT_STAGNATION_LIMIT,
                 perturb_k_destroy_percent=DEFAULT_PERTURB_K_DESTROY_PERCENT,
                 add_boundary_jobs_to_destroy=DEFAULT_ADD_BOUNDARY_JOBS,  # New parameter
                 num_boundary_jobs_to_add=DEFAULT_NUM_BOUNDARY_JOBS_TO_ADD,  # New parameter
                 verbose_level=logging.INFO):

        self.processing_times = np.array(processing_times)
        self.num_jobs, self.num_machines = self.processing_times.shape
        self.all_job_indices = list(range(self.num_jobs))

        self.ig_iterations_config = ig_iterations
        self.k_destroy_percent = k_destroy_percent
        self.k_destroy_num = 0

        self.ibs_beam_width_recon = ibs_beam_width_recon

        self.guide_functions_handler = GuideFunctions(self.processing_times, self.num_jobs, self.num_machines)
        self.guide_choice_recon = guide_choice_recon

        self.global_time_limit_config = global_time_limit
        self.global_start_time = None

        self.max_discrepancies_recon = max_discrepancies_recon
        self.max_children_to_consider_for_lds_recon = max_children_lds_recon

        self.destruction_method = destruction_method
        self.apply_local_search_on_best = apply_local_search_on_best

        self.stagnation_limit = stagnation_limit
        self.perturb_k_destroy_percent = perturb_k_destroy_percent
        self.stagnation_counter = 0

        # --- Boundary Job Parameters ---
        self.add_boundary_jobs_to_destroy = add_boundary_jobs_to_destroy
        self.num_boundary_jobs_to_add = num_boundary_jobs_to_add

        self.best_solution_overall = None
        self.best_makespan_overall = float('inf')
        self.neh_makespan_reference = float('inf')
        self.neh_sequence_reference = []

        self.current_solution_ig = []
        self.current_makespan_ig = float('inf')

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbose_level)

    # ... (_get_guide_function_for_recon_adapter and _dispatch_destroy_solution as before) ...
    def _get_guide_function_for_recon_adapter(self, alpha_for_walpha=0.5):
        if self.guide_choice_recon == 'makespan_bound_fwd':
            return self.guide_functions_handler.guide_makespan_bound_forward
        elif self.guide_choice_recon == 'walpha_fwd':
            return lambda na: self.guide_functions_handler.guide_walpha_forward(na, alpha_val=alpha_for_walpha)
        else:
            self.logger.warning(
                f"Unknown reconstruction guide: {self.guide_choice_recon}. Defaulting to makespan_bound_fwd.")
            return self.guide_functions_handler.guide_makespan_bound_forward

    def _dispatch_destroy_solution(self, current_sequence_list, effective_k_destroy, is_perturbation=False):
        destruction_type_for_log = "Perturbation" if is_perturbation else self.destruction_method.capitalize()
        if self.destruction_method == 'shaw' or (is_perturbation and self.destruction_method == 'shaw'):
            self.logger.debug(f"Using Shaw {destruction_type_for_log} (k={effective_k_destroy})")
            return destroy_solution_shaw(current_sequence_list, effective_k_destroy, self.processing_times,
                                         self.num_jobs)
        elif self.destruction_method == 'block' or is_perturbation:
            self.logger.debug(f"Using Block {destruction_type_for_log} (k={effective_k_destroy})")
            return destroy_solution_block(current_sequence_list, effective_k_destroy)
        else:
            self.logger.warning(f"Unknown destruction scenario. Defaulting to block.")
            return destroy_solution_block(current_sequence_list, effective_k_destroy)

    def solve(self):
        self.global_start_time = time.time()
        self.current_solution_ig = []
        self.current_makespan_ig = float('inf')
        self.stagnation_counter = 0

        self.logger.info(f"Starting IG-IBS Solver. Jobs: {self.num_jobs}, Machines: {self.num_machines}")
        param_log_str = (f"Params: IG_Iter={self.ig_iterations_config}, K_Destroy%={self.k_destroy_percent * 100:.1f}, "
                         f"IBS_BeamW={self.ibs_beam_width_recon}, Max_Disc_Recon={self.max_discrepancies_recon}, "
                         f"Max_Child_LDS={self.max_children_to_consider_for_lds_recon}, "
                         f"Destruction={self.destruction_method}, ApplyLS={self.apply_local_search_on_best}, ")
        param_log_str += f"StagnationLimit={self.stagnation_limit}, PerturbK%={self.perturb_k_destroy_percent * 100:.1f}, "
        # --- Log new boundary job params ---
        param_log_str += f"AddBoundaryJobs={self.add_boundary_jobs_to_destroy}, NumBoundary={self.num_boundary_jobs_to_add if self.add_boundary_jobs_to_destroy else 'N/A'}, "
        param_log_str += f"TimeLimit={self.global_time_limit_config}s, Guide={self.guide_choice_recon}"
        self.logger.info(param_log_str)

        self.k_destroy_num = max(1, int(self.num_jobs * self.k_destroy_percent))
        if self.num_jobs <= 1: self.k_destroy_num = 0
        if self.num_jobs > 1 and self.k_destroy_num >= self.num_jobs:
            self.k_destroy_num = self.num_jobs - 1
        self.logger.info(f"Normal k_destroy_num: {self.k_destroy_num}")

        # ... (NEH and initial LS logic as before) ...
        initial_sequence, initial_makespan = neh_heuristic(self.processing_times, self.num_jobs, self.num_machines)
        if not initial_sequence:
            self.logger.error("NEH failed to produce an initial solution.")
            return [], float('inf')

        self.neh_sequence_reference = list(initial_sequence)
        self.neh_makespan_reference = initial_makespan
        self.logger.info(f"Initial NEH solution: Makespan {initial_makespan:.2f}")

        if self.apply_local_search_on_best:
            self.logger.info("Applying Local Search to initial NEH solution...")
            initial_sequence_ls, initial_makespan_ls = local_search_insertion(
                self.processing_times, initial_sequence, self.num_jobs, self.num_machines
            )
            if initial_makespan_ls < initial_makespan:
                self.logger.info(f"NEH improved by LS: {initial_makespan:.2f} -> {initial_makespan_ls:.2f}")
                initial_sequence = initial_sequence_ls
                initial_makespan = initial_makespan_ls
            else:
                self.logger.info(f"LS did not improve initial NEH ({initial_makespan_ls:.2f}).")

        self.current_solution_ig = list(initial_sequence)
        self.current_makespan_ig = initial_makespan
        self.best_solution_overall = list(initial_sequence)
        self.best_makespan_overall = initial_makespan
        iterations_performed = 0

        for ig_iter in range(self.ig_iterations_config):
            iterations_performed += 1
            elapsed_time = time.time() - self.global_start_time
            if elapsed_time > self.global_time_limit_config:
                self.logger.info(
                    f"Global time limit {self.global_time_limit_config}s reached after {ig_iter} iterations.")
                break

            self.logger.info(
                f"\n--- IG Iteration {ig_iter + 1}/{self.ig_iterations_config} (Actual: {iterations_performed}) ---")
            self.logger.info(
                f"Time: {elapsed_time:.1f}s. Current IG M: {self.current_makespan_ig:.2f}. Overall Best M: {self.best_makespan_overall:.2f}. Stagnation: {self.stagnation_counter}/{self.stagnation_limit}")

            current_sequence_list_for_destroy = list(self.current_solution_ig)
            num_jobs_total_in_current = len(current_sequence_list_for_destroy)

            is_perturbation_move = False
            effective_k_destroy_for_this_iter = 0  # Initialize

            if self.stagnation_counter >= self.stagnation_limit and self.stagnation_limit > 0:
                self.logger.info(f"STAGNATION: Limit reached ({self.stagnation_limit}). Applying strong perturbation.")
                is_perturbation_move = True
                effective_k_destroy_for_this_iter = max(1, int(self.num_jobs * self.perturb_k_destroy_percent))
                # Perturb from current_solution_ig
                self.logger.info(
                    f"Perturbing from current_solution_ig (M={self.current_makespan_ig:.2f}) with k_destroy_percent={self.perturb_k_destroy_percent * 100:.1f}%")
                self.stagnation_counter = 0
            else:
                effective_k_destroy_for_this_iter = self.k_destroy_num  # Use normal k_destroy_num

            # Ensure effective_k_destroy is valid for the sequence length
            effective_k_destroy_for_this_iter = min(effective_k_destroy_for_this_iter,
                                                    num_jobs_total_in_current - 1 if num_jobs_total_in_current > 0 else 0)
            effective_k_destroy_for_this_iter = max(0, effective_k_destroy_for_this_iter)

            if not current_sequence_list_for_destroy or effective_k_destroy_for_this_iter == 0:
                self.logger.warning(
                    f"Skipping D&R in IG iter {ig_iter + 1} due to no current solution or effective_k_destroy=0.")
                if not is_perturbation_move: self.stagnation_counter += 1
                continue

            current_makespan_before_destroy = self.current_makespan_ig

            prefix, removed, suffix = self._dispatch_destroy_solution(current_sequence_list_for_destroy,
                                                                      effective_k_destroy_for_this_iter,
                                                                      is_perturbation=is_perturbation_move)

            if not removed:  # Should only happen if effective_k_destroy was 0 and somehow passed previous check
                self.logger.info("Destruction method returned no removed jobs. Continuing IG loop.")
                if not is_perturbation_move: self.stagnation_counter += 1
                continue

            # --- Add Boundary Jobs to Destruction if enabled and not a perturbation move ---
            # We typically don't want to add more jobs if it's already a large perturbation
            if self.add_boundary_jobs_to_destroy and not is_perturbation_move and len(removed) > 0:
                original_removed_count = len(removed)
                mutable_removed = list(removed)  # Ensure it's a list
                mutable_prefix = list(prefix)
                mutable_suffix = list(suffix)

                jobs_taken_from_prefix = 0
                for _ in range(self.num_boundary_jobs_to_add):
                    if mutable_prefix:
                        mutable_removed.append(mutable_prefix.pop())
                        jobs_taken_from_prefix += 1
                    else:
                        break

                jobs_taken_from_suffix = 0
                for _ in range(self.num_boundary_jobs_to_add):
                    if mutable_suffix:
                        mutable_removed.append(mutable_suffix.pop(0))
                        jobs_taken_from_suffix += 1
                    else:
                        break

                if jobs_taken_from_prefix > 0 or jobs_taken_from_suffix > 0:
                    self.logger.debug(
                        f"Added boundary jobs. From Prefix: {jobs_taken_from_prefix}, From Suffix: {jobs_taken_from_suffix}. Total removed: {len(mutable_removed)}")
                    removed = mutable_removed  # Update with new list of removed jobs
                    prefix = mutable_prefix  # Update prefix
                    suffix = mutable_suffix  # Update suffix
            # --- End of Add Boundary Jobs ---

            self.logger.debug(
                f"Reconstructing. Prefix len: {len(prefix)}, Actual Removed count: {len(removed)}, Suffix len: {len(suffix)}")

            reconstructed_seq, reconstructed_m = reconstruct_solution_ibs(
                prefix, removed, suffix,
                self.processing_times, self.num_jobs, self.num_machines,
                self.guide_functions_handler,
                self._get_guide_function_for_recon_adapter,
                self.ibs_beam_width_recon,
                self.max_discrepancies_recon,
                self.max_children_to_consider_for_lds_recon
            )

            new_overall_best_this_iter = False
            if reconstructed_seq is not None:
                log_suffix = "(Perturbation Result)" if is_perturbation_move else ""
                self.logger.info(
                    f"Reconstruction complete. New Makespan: {reconstructed_m:.2f} (Was: {current_makespan_before_destroy:.2f}) {log_suffix}")

                accepted_solution_for_next_iter = False
                if is_perturbation_move:
                    self.logger.info(f"Accepting perturbed solution (M={reconstructed_m:.2f}) to diversify search.")
                    self.current_solution_ig = list(reconstructed_seq)
                    self.current_makespan_ig = reconstructed_m
                    accepted_solution_for_next_iter = True
                elif reconstructed_m < self.current_makespan_ig:
                    self.current_solution_ig = list(reconstructed_seq)
                    self.current_makespan_ig = reconstructed_m
                    accepted_solution_for_next_iter = True
                    self.logger.info(
                        f"Accepted BETTER solution. New current_makespan_ig: {self.current_makespan_ig:.2f}")
                elif reconstructed_m == self.current_makespan_ig:
                    self.current_solution_ig = list(reconstructed_seq)
                    accepted_solution_for_next_iter = True
                    self.logger.info(f"Reconstructed makespan {reconstructed_m:.2f} is EQUAL. Accepting for diversity.")
                else:  # Worse and not a perturbation
                    self.logger.info(
                        f"Reconstructed makespan {reconstructed_m:.2f} is WORSE than current {self.current_makespan_ig:.2f}. Solution rejected.")

                # Update overall best if current_makespan_ig (which might have been updated) is better
                if self.current_makespan_ig < self.best_makespan_overall:
                    self.best_solution_overall = list(self.current_solution_ig)
                    self.best_makespan_overall = self.current_makespan_ig
                    self.logger.info(f"!!! New OVERALL BEST solution found: {self.best_makespan_overall:.2f} !!!")
                    self.stagnation_counter = 0  # Reset on new overall best
                    new_overall_best_this_iter = True
            else:
                self.logger.warning("Reconstruction failed to produce a valid sequence.")

            if not new_overall_best_this_iter and not is_perturbation_move:
                self.stagnation_counter += 1

        # ... (Final LS and Logging as before) ...
        if self.apply_local_search_on_best and self.best_solution_overall:
            self.logger.info("\n--- Applying Final Local Search to Best IG Solution ---")
            current_best_before_final_ls = self.best_makespan_overall
            final_ls_seq, final_ls_m = local_search_insertion(
                self.processing_times, self.best_solution_overall, self.num_jobs, self.num_machines
            )
            if final_ls_m < current_best_before_final_ls:
                self.logger.info(
                    f"!!! Final LS IMPROVED overall best: {current_best_before_final_ls:.2f} -> {final_ls_m:.2f} !!!")
                self.best_solution_overall = list(final_ls_seq)
                self.best_makespan_overall = final_ls_m
            else:
                self.logger.info(
                    f"Final LS did not improve overall best (LS M: {final_ls_m:.2f}, Prev Best: {current_best_before_final_ls:.2f}).")

        final_elapsed_time = time.time() - self.global_start_time
        self.logger.info("\n--- IG-IBS Solver Finished ---")
        self.logger.info(f"Total IG iterations performed: {iterations_performed}")
        self.logger.info(f"Total time: {final_elapsed_time:.2f} seconds.")
        self.logger.info(f"NEH reference makespan: {self.neh_makespan_reference:.2f}")
        self.logger.info(f"Best makespan found by IG-IBS (after any final LS): {self.best_makespan_overall:.2f}")
        if self.best_solution_overall:
            self.logger.info(f"Best sequence length: {len(self.best_solution_overall)}")

        return self.best_solution_overall, self.best_makespan_overall