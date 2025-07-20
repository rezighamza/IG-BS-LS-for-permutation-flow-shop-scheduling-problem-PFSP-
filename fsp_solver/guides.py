import numpy as np
import logging


class IBSNode:
    def __init__(self, scheduled_sequence, unscheduled_jobs, front_completion_times,
                 guide_value, depth, discrepancies_used=0):  # Added discrepancies_used
        self.scheduled_sequence = scheduled_sequence
        self.unscheduled_jobs = unscheduled_jobs
        self.front_completion_times = front_completion_times
        self.guide_value = guide_value
        self.depth = depth
        self.discrepancies_used = discrepancies_used

    def __lt__(self, other):
        if self.guide_value != other.guide_value:
            return self.guide_value < other.guide_value
        if self.discrepancies_used != other.discrepancies_used:  # New tie-breaker
            return self.discrepancies_used < other.discrepancies_used
        return self.depth > other.depth

    def get_attributes_for_guide(self):
        return {
            "scheduled_sequence": self.scheduled_sequence,
            "unscheduled_jobs": self.unscheduled_jobs,
            "front_completion_times": self.front_completion_times,
            "depth": self.depth,
        }


# --- Guide Functions ---
class GuideFunctions:
    def __init__(self, processing_times, num_jobs, num_machines):
        self.processing_times = processing_times
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self._remaining_processing_time_cache = {}

    def _clear_caches(self):
        self._remaining_processing_time_cache.clear()

    def _get_remaining_processing_times_on_machine(self, unscheduled_jobs_tuple, machine_index):
        cache_key = (unscheduled_jobs_tuple, machine_index)
        if cache_key in self._remaining_processing_time_cache:
            return self._remaining_processing_time_cache[cache_key]

        if not unscheduled_jobs_tuple:
            remaining_time = 0
        else:
            valid_jobs = [job for job in unscheduled_jobs_tuple if job < self.num_jobs]
            if not valid_jobs:
                remaining_time = 0
            else:
                remaining_time = np.sum(self.processing_times[list(valid_jobs), machine_index])

        self._remaining_processing_time_cache[cache_key] = remaining_time
        return remaining_time

    def guide_makespan_bound_forward(self, node_attrs):
        front_times = node_attrs['front_completion_times']
        unscheduled_jobs = node_attrs['unscheduled_jobs']
        current_makespan_on_last_machine = front_times[self.num_machines - 1]
        unscheduled_tuple = tuple(sorted(list(unscheduled_jobs)))
        remaining_processing_on_last_machine = self._get_remaining_processing_times_on_machine(
            unscheduled_tuple, self.num_machines - 1
        )
        lower_bound = current_makespan_on_last_machine + remaining_processing_on_last_machine
        return lower_bound

    def guide_walpha_forward(self, node_attrs,
                             alpha_val=0.5):
        current_total_depth = node_attrs['depth']

        if current_total_depth == self.num_jobs:
            return node_attrs['front_completion_times'][self.num_machines - 1]
        bound = self.guide_makespan_bound_forward(node_attrs)
        if abs(alpha_val - 1.0) < 1e-6:
            return bound
        if current_total_depth == 0 and alpha_val < 1e-6:
            return bound

        front_completion_times = node_attrs['front_completion_times']
        scheduled_sequence = node_attrs['scheduled_sequence']

        total_idle_time = 0
        if scheduled_sequence:
            sum_proc_times_per_machine = np.sum(self.processing_times[scheduled_sequence, :], axis=0)
            idle_per_machine = np.maximum(0, front_completion_times - sum_proc_times_per_machine)
            total_idle_time = np.sum(idle_per_machine)

        C_factor = self.num_jobs / self.num_machines if self.num_machines > 0 else 1.0
        guide_result = alpha_val * bound + (1 - alpha_val) * C_factor * total_idle_time

        return guide_result