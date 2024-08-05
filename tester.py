import numpy as np
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
from PSO_class_with_animation import PSO, rosenbrock_function, rastrigin_function  # Assuming the PSO class and rosenbrock_function are in pso.py
from pyswarm import pso


class PSOExperiment:
    def __init__(self, func, bounds, global_min, param_ranges, tol, num_of_stable_iterations, num_runs=10, max_iter=10):
        self.func = func
        self.bounds = bounds
        self.global_min = global_min
        self.param_ranges = param_ranges
        self.num_runs = num_runs
        self.max_iter = max_iter
        self.results = []
        self.output_dir = self.create_output_dir()
        self.tol = tol
        self.num_of_stable_iterations = num_of_stable_iterations

    def create_output_dir(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/results_{now}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def run_experiments(self):
        total_number_of_runs = 1
        for param in self.param_ranges.keys():
            total_number_of_runs = total_number_of_runs * len(self.param_ranges[param])
        current_run = 0
        for drop in self.param_ranges['drop']:
            for num_particles in self.param_ranges['num_particles']:
                for w in self.param_ranges['w']:
                    for c1 in self.param_ranges['c1']:
                        for c2 in self.param_ranges['c2']:
                            for drop_rate in self.param_ranges['drop_rate']:
                                for drop_percentage in self.param_ranges['drop_percentage']:
                                    current_run += 1
                                    print(f"Running experiment {current_run}/{total_number_of_runs}")
                                    self.run_single_experiment(num_particles, w, c1, c2, drop, drop_rate, drop_percentage)

        self.analyze_results()
        self.visualize_results()


    def run_single_experiment(self, initial_num_particles, w, c1, c2, drop, drop_rate=1.6, drop_percentage=0.8):
        error_distances = []
        exec_times = []
        convergence_iterations = []
        iteration_errors_all_runs = []
        error_scores = []

        # For storing library PSO results
        lib_error_distances = []
        lib_exec_times = []
        lib_convergence_iterations = []
        lib_error_scores = []

        for run in range(self.num_runs):
            # Custom PSO
            if self.num_runs > 10:
                print(f"Run {run} of {self.num_runs}", end='\r')
            pso_custom = PSO(self.func, self.bounds, self.global_min, initial_num_particles, w, c1, c2,
                             max_iter=self.max_iter, drop=drop, drop_rate=drop_rate, drop_percentage=drop_percentage)
            start_time = time.time()
            pso_custom.run()
            exec_time = time.time() - start_time
            error_distance = np.linalg.norm(np.array(pso_custom.global_best_position) - np.array(self.global_min))
            error_score = np.linalg.norm(pso_custom.global_best_score - self.func(pso_custom.global_best_position))

            error_distances.append(error_distance)
            exec_times.append(exec_time)
            convergence_iterations.append(pso_custom.iterations)
            iteration_errors_all_runs.append(pso_custom.iteration_errors)
            error_scores.append(error_score)

            # Library PSO
            start_time = time.time()
            # best_position, best_score = pso(self.func, [b[0] for b in self.bounds], [b[1] for b in self.bounds],
            #                                 swarmsize=initial_num_particles, maxiter=self.max_iter)
            best_position = [0] * len(self.bounds)
            best_score = 0
            exec_time = time.time() - start_time
            error_distance = np.linalg.norm(np.array(best_position) - np.array(self.global_min))

            lib_error_score = np.linalg.norm(best_score - self.func(best_position))

            lib_error_scores.append(lib_error_score)
            lib_error_distances.append(error_distance)
            lib_exec_times.append(exec_time)
            # Store the total iterations for library PSO
            lib_convergence_iterations.append(self.max_iter)

        self.results.append({
            "drop": drop,
            'initial_num_particles': initial_num_particles,
            'w': w,
            'c1': c1,
            'c2': c2,
            'drop_rate': drop_rate,
            'drop_percentage': drop_percentage,
            'error_distances': error_distances,
            'error_scores': error_scores,
            'exec_times': exec_times,
            'convergence_iterations': convergence_iterations,
            'error_scores': error_scores,
            'lib_error_distances': lib_error_distances,
            'lib_exec_times': lib_exec_times,
            'lib_convergence_iterations': lib_convergence_iterations,
            'lib_error_scores': lib_error_scores,
            'iteration_errors_all_runs': iteration_errors_all_runs

        })

    def analyze_results(self):
        for result in self.results:
            result['mean_distances'] = np.mean(result['error_distances'])
            result['std_distances'] = np.std(result['error_distances'])
            result['mean_exec_time'] = np.mean(result['exec_times'])
            result['std_exec_time'] = np.std(result['exec_times'])
            result['mean_convergence_iterations'] = np.mean(result['convergence_iterations'])
            result['std_convergence_iterations'] = np.std(result['convergence_iterations'])
            result['mean_lib_distances'] = np.mean(result['lib_error_distances'])
            result['std_lib_distances'] = np.std(result['lib_error_distances'])
            result['mean_lib_exec_time'] = np.mean(result['lib_exec_times'])
            result['std_lib_exec_time'] = np.std(result['lib_exec_times'])
            result['mean_lib_convergence_iterations'] = np.mean(result['lib_convergence_iterations'])
            result['std_lib_convergence_iterations'] = np.std(result['lib_convergence_iterations'])
            result['mean_error_scores'] = np.mean(result['error_scores'])
            result['std_error_scores'] = np.std(result['error_scores'])
            result['mean_lib_error_scores'] = np.mean(result['lib_error_scores'])
            result['std_lib_error_scores'] = np.std(result['lib_error_scores'])



    def visualize_results(self):
        results_by_params = {'initial_num_particles': {}, 'drop_rate': {}, 'drop_percentage': {}}

        for result in self.results:
            for param in results_by_params.keys():
                value = result[param]
                drop = result['drop']
                if value not in results_by_params[param]:
                    results_by_params[param][value] = {'True': [], 'False': [], 'lib': []}
                results_by_params[param][value][str(drop)].append(result)
                results_by_params[param][value]['lib'].append(result)  # Include library PSO results

        def plot_results(results, param_name):
            param_values = sorted(results.keys())
            if True in self.param_ranges['drop']:

                mean_distances_true = [np.mean([r['mean_distances'] for r in results[v]['True']]) for v in param_values]
                list_of_distances_true = [np.concatenate([r['error_distances'] for r in results[v]['True']]) for v in param_values] # New version to compute mean and std
                std_distances_true = np.std(list_of_distances_true, axis = 1) # Correct version of std_distances_true


                mean_exec_times_true = [np.mean([r['mean_exec_time'] for r in results[v]['True']]) for v in param_values]
                list_of_exec_times_true = [np.concatenate([r['exec_times'] for r in results[v]['True']]) for v in param_values]
                std_exec_times_true = np.std(list_of_exec_times_true, axis = 1)
        
            if False in self.param_ranges['drop']:
                mean_distances_false = [np.mean([r['mean_distances'] for r in results[v]['False']]) for v in param_values]
                list_of_distances_false = [np.concatenate([r['error_distances'] for r in results[v]['False']]) for v in param_values]
                std_distances_false = np.std(list_of_distances_false, axis = 1)


                mean_exec_times_false = [np.mean([r['mean_exec_time'] for r in results[v]['False']]) for v in param_values]
                list_of_exec_times_false = [np.concatenate([r['exec_times'] for r in results[v]['False']]) for v in param_values]
                std_exec_times_false = np.std(list_of_exec_times_false, axis = 1)

            
            mean_lib_distances = [np.mean([r['mean_lib_distances'] for r in results[v]['lib']]) for v in param_values]
            list_of_lib_distances = [np.concatenate([r['lib_error_distances'] for r in results[v]['lib']]) for v in param_values]
            std_lib_distances = np.std(list_of_lib_distances, axis = 1)


            mean_lib_exec_times = [np.mean([r['mean_lib_exec_time'] for r in results[v]['lib']]) for v in param_values]
            list_of_lib_exec_times = [np.concatenate([r['lib_exec_times'] for r in results[v]['lib']]) for v in param_values]
            std_lib_exec_times = np.std(list_of_lib_exec_times, axis = 1)

            # # Compute the increase in error distance
            # mean_distances_true_ = np.array(mean_distances_true)
            # mean_distances_false_ = np.array(mean_distances_false)
            # mean_distances_increase = (mean_distances_true_ ) / mean_distances_false_

            # # Compute the increase in execution time
            # mean_exec_times_true_ = np.array(mean_exec_times_true)
            # mean_exec_times_false_ = np.array(mean_exec_times_false)
            # mean_exec_times_increase = (mean_exec_times_false_ ) / mean_exec_times_true_



            # Save the results in a txt file

            # Execution timings:

            # if param_name == 'initial_num_particles':

            #     with open(os.path.join(self.output_dir, f"results_exec_times.txt"), "w") as f:
            #         f.write(f"Mean Execution Times True: {mean_exec_times_true}\n")
            #         f.write(f"Standard Deviation Execution Times True: {std_exec_times_true}\n")
            #         f.write(f"Mean Execution Times False: {mean_exec_times_false}\n")
            #         f.write(f"Standard Deviation Execution Times False: {std_exec_times_false}\n")
            #         f.write(f"Mean Execution Times Lib: {mean_lib_exec_times}\n")
            #         f.write(f"Standard Deviation Execution Times Lib: {std_lib_exec_times}\n")

            #     # Error distances:

            #     with open(os.path.join(self.output_dir, f"results_error_distances.txt"), "w") as f:
            #         f.write(f"Mean Error Distances True: {mean_distances_true}\n")
            #         f.write(f"Standard Deviation Error Distances True: {std_distances_true}\n")
            #         f.write(f"Mean Error Distances False: {mean_distances_false}\n")
            #         f.write(f"Standard Deviation Error Distances False: {std_distances_false}\n")
            #         f.write(f"Mean Error Distances Lib: {mean_lib_distances}\n")
            #         f.write(f"Standard Deviation Error Distances Lib: {std_lib_distances}\n")

            if (False in self.param_ranges['drop']) and (True in self.param_ranges['drop']) :
                if param_name == 'initial_num_particles':
                    plt.figure()
                    plt.grid(True)
                    for v in param_ranges['drop_rate']:
                        v_ = str(v)
                        for z in param_ranges['drop_percentage']:
                            z_ = str(z)
                            if v == 0:
                                plt.errorbar(param_values, mean_distances_true, #yerr=std_distances_true,
                                        fmt='o-', label='Linear Dropout',
                                        color='blue')
                            else:
                                plt.errorbar(param_values, mean_distances_true, #yerr=std_distances_true,
                                            fmt='o-', label=f'Logarithmic Dropout, Base = {v_}, k = {z_}',
                                            color='blue')
                            plt.errorbar(param_values, mean_distances_false, #yerr=std_distances_false,
                                        fmt='o-', label='Baseline',
                                        color='red')
                            # plt.errorbar(param_values, mean_lib_distances, #yerr=std_lib_distances,
                            #              fmt='o-', label='lib_pso',
                            #              color='green')
                            
                    
                    # plt.xscale('log')
                    plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.9)
                    plt.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
                    plt.minorticks_on()
                    plt.tick_params(which='minor', length=4, color='gray')
                    plt.xlabel('Initial Swarm Size (N)')
                    plt.ylabel('Mean Distance')
                    plt.title(f'Mean Distance vs. Initial Swarm Size (N)')
                    plt.legend()
        
                    plt.savefig(os.path.join(self.output_dir, f'distance_vs_swarm_size_Base_{v_}_k_{z_}.png'))
                    plt.close()


                    plt.figure()
                    plt.grid(True)
                    for v in param_ranges['drop_rate']:
                        for z in param_ranges['drop_percentage']:
                            if v == 0:
                                plt.errorbar(param_values, mean_exec_times_true, #yerr=std_exec_times_true,
                                        fmt='o-', label='Linear Dropout',
                                        color='blue')
                            else:
                                plt.errorbar(param_values, mean_exec_times_true, #yerr=std_exec_times_true,
                                            fmt='o-', label=f'Logarithmic Dropout, Base = {v}, k = {z}',
                                            color='blue')
                            plt.errorbar(param_values, mean_exec_times_false, #yerr=std_exec_times_false,
                                        fmt='o-', label='Baseline',
                                        color='red')
                            # plt.errorbar(param_values, mean_lib_exec_times, #yerr=std_lib_exec_times,
                            #              fmt='o-', label='lib_pso',
                            #              color='green')
                            
                    
                    # plt.xscale('log')
                    plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.9)
                    plt.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
                    plt.minorticks_on()
                    plt.tick_params(which='minor', length=4, color='gray')
                    plt.xlabel('Initial Swarm Size (N)')
                    plt.ylabel('Mean Execution Time (s)')
                    plt.title(f'Mean Execution Time vs. Initial Swarm Size (N)')
                    plt.legend()
                    plt.savefig(os.path.join(self.output_dir, f'exec_time_vs_swarm_size_Base_{v_}_k_{z_}.png'))
                    plt.close()

            #     plt.figure()
            #     plt.grid(True)
            #     plt.errorbar(param_values, mean_distances_increase, #yerr=std_distances_true,
            #                  fmt='o-', color='blue', label='Increase in Distance')
            #     plt.errorbar(param_values, mean_exec_times_increase, #yerr=std_exec_times_true,
            #                     fmt='o-', color='red', label='Decrease in Time')
            #    # plt.xscale('log')
            #     plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.9)
            #     plt.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
            #     plt.minorticks_on()
            #     plt.tick_params(which='minor', length=4, color='gray')
            #     plt.xlabel('Initial Swarm Size (N)')
            #     plt.ylabel('Performance Improvement')
            #     plt.title(f'Performance of Efficient PSO over Baseline for Different Initial Swarm Sizes (N)')
            #     plt.legend()
            #     plt.savefig(os.path.join(self.output_dir, f'increase_distance_vs_swarm_size.png'))
            #     plt.close()




            if param_name == 'drop_percentage':
                mean_dist_rate_true = []
                rate_str = []
                for rate in self.param_ranges['drop_rate']:
                    if rate == 0:
                        rate_str.append('Linear Dropout')
                    else:
                        rate_str.append(str(rate))
                    mean_dist_rate_true.append([np.mean([r['mean_distances'] for r in results[v]['True'] if r['drop_rate'] == rate]) for v in param_values])
                plt.figure()
                plt.grid(True)
                i=0
                for curve in mean_dist_rate_true:
                    if rate_str[i] == 'Linear Dropout':
                        plt.plot(param_values, curve,'o-', label='Linear Dropout')
                    else:
                        plt.plot(param_values, curve,'o-', label=f'Logarithmic Dropout, Base = {rate_str[i]}')
                    i+=1
                plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.9)
                plt.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
                plt.minorticks_on()
                plt.tick_params(which='minor', length=4, color='gray')
                plt.xlabel("Dropout Factor, k")
                plt.ylabel('Mean Distance')
                plt.title(f'Mean Distance vs. Different Dropout Factors')
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, f'distance_vs_different_dropout_factors_rate.png'))
                plt.close()

                mean_exec_time_rate_true = []
                for rate in self.param_ranges['drop_rate']:
                    mean_exec_time_rate_true.append([np.mean([r['mean_exec_time'] for r in results[v]['True'] if r['drop_rate'] == rate]) for v in param_values])
                plt.figure()
                plt.grid(True)
                i=0
                for curve in mean_exec_time_rate_true:
                    if rate_str[i] == 'Linear Dropout':
                        plt.plot(param_values, curve,'o-', label='Linear Dropout')
                    else:
                        plt.plot(param_values, curve,'o-', label=f'Logarithmic Dropout, Base = {rate_str[i]}')
                    i+=1
                plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.9)
                plt.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
                plt.minorticks_on()
                plt.tick_params(which='minor', length=4, color='gray')
                plt.xlabel("Dropout Factor, k")
                plt.ylabel('Mean Execution Time (s)')
                plt.title(f'Mean Execution Time vs. Different Dropout Factors')
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, f'exec_time_vs_different_dropout_factors_rate.png'))
                plt.close()



            # if param_name == 'initial_num_particles':
            #     convergence_iters_true = [np.mean([np.mean(r['convergence_iterations']) for r in results[v]['True']])
            #                               for v in param_values]
            #     convergence_iters_false = [np.mean([np.mean(r['convergence_iterations']) for r in results[v]['False']])
            #                                for v in param_values]
            #     convergence_iters_lib = [
            #         np.mean([np.mean(r['mean_lib_convergence_iterations']) for r in results[v]['lib']]) for v in
            #         param_values]

            #     plt.figure()
            #     plt.grid(True)
            #     plt.plot(param_values, convergence_iters_true, 'o-', label='drop=True', color='blue')
            #     plt.plot(param_values, convergence_iters_false, 'o-', label='drop=False', color='red')
            #     plt.plot(param_values, convergence_iters_lib, 'o-', label='lib_pso', color='green')
            #    # plt.xscale('log')
            #     plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.9)
            #     plt.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
            #     plt.minorticks_on()
            #     plt.tick_params(which='minor', length=4, color='gray')
            #     plt.xlabel('Initial Swarm Size (N)')
            #     plt.ylabel('Mean Iterations to Converge')
            #     plt.title(f'Iterations to Converge vs. Initial Swarm Size (N)')
            #     plt.legend()
            #     plt.savefig(os.path.join(self.output_dir, f'iterations_vs_swarm_size.png'))
            #     plt.close()

        for param_name, results in results_by_params.items():
            plot_results(results, param_name)

        # for result in self.results:
            # if result['initial_num_particles'] == self.param_ranges['num_particles'][0]:
                
            #     errors_true = [errors for r in self.results if r['drop'] == True for errors in
            #                    r['iteration_errors_all_runs']]
            #     errors_false = [errors for r in self.results if r['drop'] == False for errors in
            #                     r['iteration_errors_all_runs']]
            #     padded_errors_true = pad_iteration_errors(errors_true)
            #     padded_errors_false = pad_iteration_errors(errors_false)
            #     avg_errors_true = np.nanmean(padded_errors_true, axis=0)
            #     avg_errors_false = np.nanmean(padded_errors_false, axis=0)

            #     plt.figure()
            #     plt.grid(True)
            #     plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.9)
            #     plt.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
            #     plt.minorticks_on()
            #     plt.tick_params(which='minor', length=4, color='gray')
            #     plt.plot(range(len(avg_errors_true)), avg_errors_true, label='drop=True', color='blue')
            #     plt.plot(range(len(avg_errors_false)), avg_errors_false, label='drop=False', color='red')
            #     plt.xlabel('Iteration')
            #     plt.ylabel('Average Error')
            #     plt.title('Average Error per Iteration')
            #     plt.legend()
            #     plt.savefig(os.path.join(self.output_dir, 'average_error_per_iteration.png'))
            #     plt.close()
            #     break

# Define parameter ranges
param_ranges = {'drop': [True, False],
                'num_particles':  [32, 64, 128, 256, 512, 750, 1024, 1500, 2048],
                'w': [0.2],
                'c1': [1.6],
                'c2': [1.6],
                'drop_rate': [3],
                'drop_percentage': [0.9]}


def pad_iteration_errors(iteration_errors_all_runs):
    max_length = max(len(errors) for errors in iteration_errors_all_runs)
    padded_errors = [np.pad(errors, (0, max_length - len(errors)), 'constant', constant_values=np.nan) for errors in
                     iteration_errors_all_runs]
    return padded_errors


def main():

    dim = 4
    lb = [-10] * dim
    up = [10] * dim
    bounds = [[lb[i], up[i]] for i in range(dim)]

    tolerance = 1e-1
    num_of_stable_iterations = 5
    rosenbrock_global_min = [1, 1]  # True global minimum for the Rosenbrock function
    rastrigin_global_min = [0] * dim
    max_iter = 40
    num_runs = 200
    experiment = PSOExperiment(rastrigin_function, bounds, rastrigin_global_min, param_ranges, num_runs=num_runs, max_iter=max_iter,
                               tol=tolerance, num_of_stable_iterations=num_of_stable_iterations)
    experiment.run_experiments()
    output_dir = experiment.output_dir

    with open(output_dir + "/parameters.txt", "w") as f:
        f.write(f"Parameter ranges: {param_ranges}\n")
        f.write(f"Bounds: {bounds}\n")
        f.write(f"max_iter: {max_iter}\n")
        f.write(f"tolerance: {tolerance}\n")
        f.write(f"num_of_stable_iterations: {num_of_stable_iterations}\n")

    np.save(output_dir + "/results.npy", experiment.results)


if __name__ == "__main__":
    main()