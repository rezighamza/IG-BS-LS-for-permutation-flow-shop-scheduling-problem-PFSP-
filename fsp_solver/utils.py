import numpy as np
import os
import logging

def read_flow_shop_data(file_path, machine_count, job_count):
    '''
    Reading the data from the benchmark file
    '''
    instances = []
    with open(file_path) as p:
        lines = p.readlines()
        line_count = len(lines)

        instance_count = line_count // (machine_count + 3)

        for i in range(instance_count):
            # recover the data of each instance
            params_line = lines[i * (machine_count + 3) + 1]
            job_count, machine_count, initial_seed, upper_bound, lower_bound = list(
                map(lambda x: int(x), params_line.split()))

            processing_times = np.array([list(map(lambda x: int(x), line.strip().split())) for
                                         line in lines[
                                                 i * (machine_count + 3) + 3:  # start
                                                 i * (machine_count + 3) + 3 + machine_count  # end
                                                 ]
                                         ])

            record = {
                "instance_id": i + 1,
                "machine_count": machine_count,
                "job_count": job_count,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "processing_times": processing_times.T
            }
            instances.append(record)

    return instances


# In your utils.py
def evaluate_sequence(processing_times_NxM, sequence_indices, num_machines):  # Renamed for clarity
    if not sequence_indices:  # Handle empty sequence
        return 0, np.zeros(num_machines)

    num_seq_jobs = len(sequence_indices)  # Use sequence_indices
    # completion_times[job_idx_in_sequence][machine_idx]
    completion_times_matrix = np.zeros((num_seq_jobs, num_machines))

    # First job in the sequence
    first_job_actual_idx = sequence_indices[0]
    completion_times_matrix[0][0] = processing_times_NxM[first_job_actual_idx][0]
    for m_idx in range(1, num_machines):  # Corrected loop for machines
        completion_times_matrix[0][m_idx] = completion_times_matrix[0][m_idx - 1] + \
                                            processing_times_NxM[first_job_actual_idx][m_idx]

    # Subsequent jobs in the sequence
    for s_idx in range(1, num_seq_jobs):  # s_idx is the index in the current sequence
        job_actual_idx = sequence_indices[s_idx]
        # First machine for this job
        completion_times_matrix[s_idx][0] = completion_times_matrix[s_idx - 1][0] + \
                                            processing_times_NxM[job_actual_idx][0]
        # Subsequent machines for this job
        for m_idx in range(1, num_machines):
            completion_times_matrix[s_idx][m_idx] = max(completion_times_matrix[s_idx - 1][m_idx],
                                                        completion_times_matrix[s_idx][m_idx - 1]) + \
                                                    processing_times_NxM[job_actual_idx][m_idx]

    makespan = completion_times_matrix[num_seq_jobs - 1][num_machines - 1]
    last_job_completion_times_vector = completion_times_matrix[num_seq_jobs - 1, :] if num_seq_jobs > 0 else np.zeros(
        num_machines)
    return makespan, last_job_completion_times_vector


def setup_logging(log_file_path='logs/experiment_log.log', level=logging.INFO):
    """Sets up logging to file and console."""
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(level=level,
                        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path, mode='w'), # Overwrite log file each run
                            logging.StreamHandler()
                        ])
    logging.info("Logging setup complete.")