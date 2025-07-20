import os
import logging
# from fsp_solver import IG_IBS_Solver, read_flow_shop_data, setup_logging
from fsp_solver.solver_core import IG_IBS_Solver
from fsp_solver.utils import read_flow_shop_data, setup_logging

from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args

# --- Global Settings for skopt ---
INSTANCE_FILENAME = 'tai100_10.txt'  # Example instance file
EXPECTED_JOBS, EXPECTED_MACHINES = 100 , 10

N_CALLS_SKOPT = 120  # More calls for more params
SOLVER_TIME_LIMIT_S = 30  # Slightly more time
SOLVER_IG_ITERS = 1000  # Higher cap

# --- Define Parameter Space for skopt ---
param_space = [
    Integer(15, 40, name='k_destroy_p_scaled'),
    Integer(10, 35, name='ibs_beam_w'),  # Slightly wider range
    Categorical(['walpha_fwd', 'makespan_bound_fwd'], name='guide_choice'),
    Integer(0, 2, name='max_discrepancies'),
    Integer(2, 6, name='max_children_lds'),  # Up to 6
    Categorical(['block', 'shaw'], name='destruction_method'),
    Categorical([True, False], name='apply_ls'),
    Integer(20, 100, name='stagnation_limit'),
    Integer(40, 80, name='perturb_k_destroy_p_scaled'),
    Categorical([True, False], name='add_boundary_jobs'),  # New
    Integer(1, 2, name='num_boundary_jobs')  # New (1 or 2 from each side)
    # Only used if add_boundary_jobs is True
]

# ... (load_instance_and_references function as before) ...
instance_data_cache = None
instance_ub_cache = None


def load_instance_and_references():
    global instance_data_cache, instance_ub_cache
    if instance_data_cache is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(current_dir, 'data')
        instance_filepath = os.path.join(file_dir, INSTANCE_FILENAME)

        all_instances_data = read_flow_shop_data(instance_filepath,
                                                 EXPECTED_MACHINES,
                                                 EXPECTED_JOBS)
        if not all_instances_data:
            raise ValueError(f"Could not load instance data for {INSTANCE_FILENAME}")
        instance_data_cache = all_instances_data[0]
        instance_ub_cache = instance_data_cache.get('upper_bound', None)

    return instance_data_cache['processing_times'], instance_ub_cache


@use_named_args(param_space)
def objective_function(k_destroy_p_scaled, ibs_beam_w, guide_choice,
                       max_discrepancies, max_children_lds,
                       destruction_method, apply_ls,
                       stagnation_limit, perturb_k_destroy_p_scaled,
                       add_boundary_jobs, num_boundary_jobs):  # Added new params

    k_destroy_p = k_destroy_p_scaled / 100.0
    perturb_k_p = perturb_k_destroy_p_scaled / 100.0

    processing_times, _ = load_instance_and_references()

    # If add_boundary_jobs is False, num_boundary_jobs is irrelevant,
    # but skopt will still pass a value. The solver will ignore it internally.

    solver = IG_IBS_Solver(
        processing_times=processing_times,
        ig_iterations=SOLVER_IG_ITERS,
        k_destroy_percent=k_destroy_p,
        ibs_beam_width_recon=ibs_beam_w,
        guide_choice_recon=guide_choice,
        global_time_limit=SOLVER_TIME_LIMIT_S,
        max_discrepancies_recon=max_discrepancies,
        max_children_lds_recon=max_children_lds,
        destruction_method=destruction_method,
        apply_local_search_on_best=apply_ls,
        stagnation_limit=stagnation_limit,
        perturb_k_destroy_percent=perturb_k_p,
        add_boundary_jobs_to_destroy=add_boundary_jobs,  # Pass to solver
        num_boundary_jobs_to_add=num_boundary_jobs,  # Pass to solver
        verbose_level=logging.WARNING
    )

    _, makespan = solver.solve()

    if makespan == float('inf') or makespan is None:
        makespan = 9999999

    print(f"Params: k_p={k_destroy_p:.2f}, beam_w={ibs_beam_w}, guide={guide_choice}, max_d={max_discrepancies}, "
          f"max_c_lds={max_children_lds}, dest={destruction_method}, ls={apply_ls}, "
          f"stag_lim={stagnation_limit}, pert_k={perturb_k_p:.2f}, "
          f"add_bj={add_boundary_jobs}, num_bj={num_boundary_jobs if add_boundary_jobs else 'N/A'} "
          f"-> Makespan: {makespan:.2f}")
    return makespan


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, 'logs')
    skopt_log_filename = f"skopt_tune_phase2_stagnation_boundary_{os.path.splitext(INSTANCE_FILENAME)[0]}.log"
    skopt_log_filepath = os.path.join(log_dir, skopt_log_filename)
    setup_logging(log_file_path=skopt_log_filepath, level=logging.INFO)

    logging.info(
        f"Starting skopt hyperparameter optimization for {INSTANCE_FILENAME} (with Stagnation & Boundary Jobs)")
    _, instance_ub = load_instance_and_references()

    if instance_ub is not None:
        logging.info(f"Instance UB for RPD calculation: {instance_ub}")
    else:
        logging.info("Instance UB not found, RPD will be N/A.")

    logging.info(f"Number of calls: {N_CALLS_SKOPT}")
    logging.info(f"Solver time limit per call: {SOLVER_TIME_LIMIT_S}s")
    logging.info(f"Parameter space dimensions: {len(param_space)}")

    result = gp_minimize(
        func=objective_function,
        dimensions=param_space,
        n_calls=N_CALLS_SKOPT,
        n_initial_points=30,  # More initial points for 11 dimensions
        random_state=222,  # New random state
    )

    logging.info("\n--- skopt Optimization Finished ---")
    best_makespan_skopt = result.fun
    logging.info(f"Best makespan found by skopt: {best_makespan_skopt:.2f}")

    if instance_ub is not None and instance_ub > 0:
        rpd_perc_skopt = ((best_makespan_skopt - instance_ub) / instance_ub) * 100
        logging.info(f"RPD from UB ({instance_ub}) for skopt's best: {rpd_perc_skopt:.2f}%")
    else:
        logging.info("RPD from UB for skopt's best: N/A")

    best_params_names = [dim.name for dim in param_space]
    best_params_values_raw = result.x
    best_params_dict = dict(zip(best_params_names, best_params_values_raw))

    # Rescale scaled parameters
    if 'k_destroy_p_scaled' in best_params_dict:
        best_params_dict['k_destroy_p'] = best_params_dict['k_destroy_p_scaled'] / 100.0
        del best_params_dict['k_destroy_p_scaled']
    if 'perturb_k_destroy_p_scaled' in best_params_dict:
        best_params_dict['perturb_k_destroy_p'] = best_params_dict['perturb_k_destroy_p_scaled'] / 100.0
        del best_params_dict['perturb_k_destroy_p_scaled']

    logging.info("Best parameters found by skopt:")
    for name, value in best_params_dict.items():
        if name == 'num_boundary_jobs' and not best_params_dict.get('add_boundary_jobs'):
            logging.info(f"  {name}: {value} (N/A as add_boundary_jobs is False)")
        elif isinstance(value, float):
            logging.info(f"  {name}: {value:.4f}")
        else:
            logging.info(f"  {name}: {value}")

    # Optional plotting code (as before)