import os
import logging
import time  # For individual instance timing
import pandas as pd  # For creating a nice summary table
from fsp_solver import IG_IBS_Solver, read_flow_shop_data, setup_logging

# --- Default parameters for the solver if not overridden by benchmark-specific ones ---
# These can be adjusted based on general skopt findings or as a fallback
DEFAULT_K_DESTROY_P = 0.30
DEFAULT_PERTURB_K_DESTROY_P = 0.50  # Assuming you have this parameter in your solver
DEFAULT_IBS_BEAM_W = 20
DEFAULT_GUIDE_CHOICE = 'walpha_fwd'  # Default, can be overridden
DEFAULT_MAX_DISCREPANCIES = 1
DEFAULT_MAX_CHILDREN_LDS = 3
DEFAULT_DESTRUCTION_METHOD = 'block'
DEFAULT_APPLY_LS = True
DEFAULT_STAGNATION_LIMIT = 50  # Assuming you have this parameter
DEFAULT_ADD_BOUNDARY_JOBS = False  # Assuming you have this
DEFAULT_NUM_BOUNDARY_JOBS = 1  # Assuming you have this


def run_benchmark_set(
        benchmark_filename_template,  # e.g., "tai{jobs}_{machines}.txt"
        specific_filename_to_run,  # e.g., "tai50_10.txt" - the actual file containing 10 instances
        jobs_in_instance,
        machines_in_instance,
        log_level=logging.INFO,
        ig_iters_cap=1000,  # Max iterations if time limit not hit
        time_limit_per_instance_s=60,
        # Pass skopt-tuned or default parameters
        k_destroy_p=DEFAULT_K_DESTROY_P,
        perturb_k_destroy_p=DEFAULT_PERTURB_K_DESTROY_P,
        ibs_beam_w=DEFAULT_IBS_BEAM_W,
        guide_choice=DEFAULT_GUIDE_CHOICE,
        max_discrepancies=DEFAULT_MAX_DISCREPANCIES,
        max_children_lds=DEFAULT_MAX_CHILDREN_LDS,
        destruction_method=DEFAULT_DESTRUCTION_METHOD,
        apply_ls=DEFAULT_APPLY_LS,
        stagnation_limit=DEFAULT_STAGNATION_LIMIT,
        add_boundary_jobs=DEFAULT_ADD_BOUNDARY_JOBS,
        num_boundary_jobs=DEFAULT_NUM_BOUNDARY_JOBS
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(current_dir, 'data')
    log_dir = os.path.join(current_dir, 'logs')

    # General log for the whole benchmark set run
    # Extracting base name like "tai50_10" from "tai50_10.txt"
    base_instance_name = os.path.splitext(specific_filename_to_run)[0]
    # Suffix for log file indicating it's a multi-instance run with new features
    log_file_suffix = "_all_instances_stagnation_boundary.log"  # As requested

    benchmark_log_filename = f"{base_instance_name}{log_file_suffix}"
    benchmark_log_filepath = os.path.join(log_dir, benchmark_log_filename)

    # Setup one main log file for the entire benchmark run
    # Individual solver runs will also log to this if their logger is configured appropriately
    # or if we rely on root logger propagation.
    # For cleaner separation, solver could log to its own temp file or be less verbose.
    # For now, let's make solver less verbose during multi-instance runs.
    setup_logging(log_file_path=benchmark_log_filepath, level=log_level)  # Main log for this script

    solver_verbose_level = logging.WARNING if log_level <= logging.INFO else log_level  # Make solver quieter

    logging.info(f"--- Starting Benchmark Run for File: {specific_filename_to_run} ---")
    logging.info(f"Job Count: {jobs_in_instance}, Machine Count: {machines_in_instance}")
    logging.info(f"Time Limit per Instance: {time_limit_per_instance_s}s")
    logging.info(
        f"Parameters for this run: k_p={k_destroy_p:.2f}, perturb_k%={perturb_k_destroy_p:.2f}, beam_w={ibs_beam_w}, guide={guide_choice}, "
        f"max_d={max_discrepancies}, max_c_lds={max_children_lds}, dest={destruction_method}, ls={apply_ls}, "
        f"stag_lim={stagnation_limit}, add_bound={add_boundary_jobs}, num_bound={num_boundary_jobs}")

    instance_filepath = os.path.join(file_dir, specific_filename_to_run)
    all_instances_data_from_file = read_flow_shop_data(instance_filepath,
                                                       machines_in_instance,
                                                       jobs_in_instance)

    if not all_instances_data_from_file:
        logging.error(f"No instance data loaded from {specific_filename_to_run}. Exiting.")
        return

    results_summary = []
    total_benchmark_time_start = time.time()

    for idx, instance_data in enumerate(all_instances_data_from_file):
        instance_id_in_file = instance_data['instance_id']  # This is 1-based from your parser
        # Taillard instance names are usually like ta001, ta002 for 20x5; ta021, ta022 for 50x5 etc.
        # We need a mapping if we want the exact "taXXX" name.
        # For now, use "BenchmarkFile_InstanceInFile_X"
        display_instance_name = f"{base_instance_name}_Inst{instance_id_in_file}"

        logging.info(f"\n--- Solving: {display_instance_name} ({idx + 1}/{len(all_instances_data_from_file)}) ---")

        processing_times = instance_data['processing_times']
        instance_ub = instance_data.get('upper_bound', None)
        instance_lb = instance_data.get('lower_bound', None)  # Not used for RPD, but good to have

        if instance_ub is not None:
            logging.info(f"Instance UB: {instance_ub}, LB: {instance_lb}")

        instance_run_start_time = time.time()

        # Assuming your solver's __init__ now accepts all these new parameters
        solver = IG_IBS_Solver(
            processing_times=processing_times,
            ig_iterations=ig_iters_cap,
            k_destroy_percent=k_destroy_p,
            # perturb_k_destroy_percent=perturb_k_destroy_p, # Add if solver supports
            # stagnation_limit=stagnation_limit,             # Add if solver supports
            # add_boundary_jobs_to_destruction=add_boundary_jobs, # Add if solver supports
            # num_boundary_jobs=num_boundary_jobs,               # Add if solver supports
            ibs_beam_width_recon=ibs_beam_w,
            guide_choice_recon=guide_choice,
            global_time_limit=time_limit_per_instance_s,  # Time limit for THIS instance
            max_discrepancies_recon=max_discrepancies,
            max_children_lds_recon=max_children_lds,
            destruction_method=destruction_method,
            apply_local_search_on_best=apply_ls,
            verbose_level=solver_verbose_level  # Make individual solver runs less chatty
        )

        # Add new parameters to solver call if your IG_IBS_Solver class supports them
        # For example, if you added `perturb_k_destroy_percent` to its __init__:
        # solver.perturb_k_destroy_percent = perturb_k_destroy_p # Or pass in __init__
        # solver.stagnation_limit = stagnation_limit             # Or pass in __init__
        # Make sure your IG_IBS_Solver __init__ signature matches all params you want to pass.
        # For now, I'll assume the current IG_IBS_Solver __init__ is used.
        # You will need to add these new parameters to IG_IBS_Solver's __init__ and logic.
        # --- Placeholder for where you'd set new params if not in __init__ ---
        # if hasattr(solver, 'stagnation_limit'): solver.stagnation_limit = stagnation_limit
        # if hasattr(solver, 'perturb_k_destroy_percent'): solver.perturb_k_destroy_percent = perturb_k_destroy_p
        # etc.

        best_sequence, best_makespan = solver.solve()
        instance_run_time = time.time() - instance_run_start_time

        neh_m = solver.neh_makespan_reference
        impr_neh = float('nan')
        rpd_ub = float('nan')

        if neh_m is not None and neh_m > 0 and neh_m != float('inf') and best_makespan != float('inf'):
            impr_neh = ((neh_m - best_makespan) / neh_m) * 100

        if instance_ub is not None and instance_ub > 0 and best_makespan != float('inf'):
            rpd_ub = ((best_makespan - instance_ub) / instance_ub) * 100

        results_summary.append({
            "Instance": display_instance_name,
            "NEH Makespan": neh_m,
            "Solver Makespan": best_makespan if best_makespan != float('inf') else "Fail",
            "Impr. NEH (%)": f"{impr_neh:.2f}" if not pd.isna(impr_neh) else "N/A",
            "UB (Taillard)": instance_ub if instance_ub is not None else "N/A",
            "RPD UB (%)": f"{rpd_ub:.2f}" if not pd.isna(rpd_ub) else "N/A",
            "Exec Time (s)": f"{instance_run_time:.2f}"
        })
        logging.info(
            f"Finished {display_instance_name}. Solver M: {best_makespan}, NEH M: {neh_m}, RPD: {rpd_ub:.2f}%, Time: {instance_run_time:.2f}s")

    total_benchmark_time_end = time.time()
    logging.info(f"\n--- Benchmark Set Finished: {specific_filename_to_run} ---")
    logging.info(f"Total time for all instances: {total_benchmark_time_end - total_benchmark_time_start:.2f} seconds.")

    # Print summary table
    summary_df = pd.DataFrame(results_summary)
    logging.info("\n--- Results Summary Table ---")
    # Convert DataFrame to string for logging, then print line by line
    summary_string = summary_df.to_string(index=False)
    for line in summary_string.split('\n'):
        logging.info(line)

    # Calculate and log averages
    valid_rpd_values = [float(r["RPD UB (%)"]) for r in results_summary if
                        isinstance(r["RPD UB (%)"], str) and r["RPD UB (%)"] != "N/A"]
    valid_impr_values = [float(r["Impr. NEH (%)"]) for r in results_summary if
                         isinstance(r["Impr. NEH (%)"], str) and r["Impr. NEH (%)"] != "N/A"]

    if valid_rpd_values:
        avg_rpd = sum(valid_rpd_values) / len(valid_rpd_values)
        logging.info(f"\nAverage RPD from UB: {avg_rpd:.2f}%")
    if valid_impr_values:
        avg_impr_neh = sum(valid_impr_values) / len(valid_impr_values)
        logging.info(f"Average Improvement over NEH: {avg_impr_neh:.2f}%")


if __name__ == '__main__':
    # --- Select Benchmark and Parameters ---

    # Configuration for Taillard 20x5 (tai001-tai010)
    # Based on skopt for tai20_5 instance 1 (ta001)
    # benchmark_file = "tai20_5.txt"
    # jobs, machines = 20, 5
    # params = {
    #     "k_destroy_p": 0.37, "perturb_k_destroy_p": 0.80, "ibs_beam_w": 32,
    #     "guide_choice": 'walpha_fwd', "max_discrepancies": 2, "max_children_lds": 2,
    #     "destruction_method": 'block', "apply_ls": True, "stagnation_limit": 95,
    #     "add_boundary_jobs": False, "num_boundary_jobs": 1,
    #     "time_limit_per_instance_s": 60 # Short for 20x5
    # }

    # Configuration for Taillard 50x10 (tai021-tai030)
    # Based on skopt for tai50_10 instance 1 (ta021)
    # benchmark_file = "tai50_10.txt"
    # jobs, machines = 50, 10
    # params = {
    #     "k_destroy_p": 0.31, "perturb_k_destroy_p": 0.45, "ibs_beam_w": 25,
    #     "guide_choice": 'walpha_fwd', "max_discrepancies": 0, "max_children_lds": 6,  # LDS off effectively
    #     "destruction_method": 'block', "apply_ls": True, "stagnation_limit": 90,
    #     "add_boundary_jobs": False, "num_boundary_jobs": 2,
    #     "time_limit_per_instance_s": 60  # Moderate time for 50x10
    # }

    # Configuration for Taillard 100x10 (tai081-tai090)
    # Based on skopt for tai100_10 instance 1 (ta081)
    benchmark_file = "tai100_10.txt"
    jobs, machines = 100, 10
    params = {
        "k_destroy_p": 0.20, "perturb_k_destroy_p": 0.48, "ibs_beam_w": 24,
        "guide_choice": 'makespan_bound_fwd', "max_discrepancies": 1, "max_children_lds": 5,
        "destruction_method": 'shaw', "apply_ls": True, "stagnation_limit": 28,
        "add_boundary_jobs": False, "num_boundary_jobs": 1,
        "time_limit_per_instance_s": 120 # Longer time for 100x10
    }
    instance_to_run = 'tai20_5.txt'
    jobs, machines = 20, 5
    time_limit = 240
    k_p = 0.30  
    beam_w = 20
    max_d = 1
    max_c_lds = 4
    dest_m = 'shaw'
    apply_final_ls = True

    run_benchmark_set(
        benchmark_filename_template=f"tai{jobs}_{machines}.txt",  # For reference, not directly used for path
        specific_filename_to_run=benchmark_file,
        jobs_in_instance=jobs,
        machines_in_instance=machines,
        log_level=logging.INFO,  # INFO for summary, DEBUG for extreme detail
        ig_iters_cap=2000,  # High cap, time limit usually controls
        **params  # Unpack the dictionary of parameters
    )