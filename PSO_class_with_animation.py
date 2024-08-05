import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


class PSO:
    def __init__(self, func, bounds, global_min, initial_num_particles=30, w=0.5, c1=1.5, c2=1.5, tol=1e-6, max_iter=1000,
                 stable_iterations=10, drop = False, drop_rate = 1.4, drop_percentage = 0.8):
        self.func = func
        self.bounds = bounds
        self.global_min = global_min  # True global minimum
        self.initial_num_particles = initial_num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tol = tol
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])
        self.particles = np.random.uniform(low=self.lb, high=self.ub, size=(initial_num_particles, self.dim))
        self.velocities = np.random.uniform(low=-1, high=1, size=(initial_num_particles, self.dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.array([self.func(p) for p in self.particles])
        self.global_best_position = self.particles[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)
        self.iterations = 0
        self.prev_best_score = self.global_best_score
        self.stable_iterations = stable_iterations
        self.best_position_history = []
        self.drop = drop
        self.drop_rate = drop_rate
        self.drop_percentage = drop_percentage
        self.iteration_errors = []

    def update_particles(self, drop = False):
        self.iterations += 1  # Increment the iteration count


        if drop :
            #mask = np.argsort(self.personal_best_scores)[:int(len(self.particles) * self.drop_percentage)]
            mask = np.argsort(self.personal_best_scores)
            mask = mask[mask < int(len(self.particles) * self.drop_percentage)]
            self.particles = self.particles[mask]
            self.velocities = self.velocities[mask]
            self.personal_best_positions = self.personal_best_positions[mask]
            self.personal_best_scores = self.personal_best_scores[mask]


        r1 = np.random.rand(len(self.particles), self.dim)
        r2 = np.random.rand(len(self.particles), self.dim)

        cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.particles)
        social_velocity = self.c2 * r2 * (self.global_best_position - self.particles)
        self.velocities = self.w * self.velocities + cognitive_velocity + social_velocity

         # Update position
        self.particles = self.particles + self.velocities

        # Apply bounds
        self.particles = np.clip(self.particles, self.lb, self.ub)

        # Evaluate the new position
        scores = np.array([self.func(p) for p in self.particles])

        # Update personal best
        mask = scores < self.personal_best_scores
        self.personal_best_positions[mask] = self.particles[mask]
        self.personal_best_scores[mask] = scores[mask]

        # Update global best
        self.global_best_position = self.particles[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)
            

        self.iteration_errors.append(self.global_best_score)




    def has_converged(self):
        self.best_position_history.append(self.global_best_position.copy())
        if len(self.best_position_history) > self.stable_iterations:
            self.best_position_history.pop(0)

        if len(self.best_position_history) < self.stable_iterations:
            return False

        # Check if all positions in the history are close to the current global best position
        for pos in self.best_position_history:
            if not np.allclose(pos, self.global_best_position, atol=self.tol):
                return False

        return True

    def run(self):
        # Sampling exponentially
        base = self.drop_rate  # Base of the exponential function, can be adjusted
        if base == 0: # linear dropout 
            sample_times = np.linspace(0, self.max_iter, 20)
        else:
            sample_times = [int(base ** i) for i in range(int(np.log(self.max_iter) / np.log(base)) + 1)]
        sample_times = sorted(set(sample_times))
        drop =  False

        for iteration in range(self.max_iter):
            if iteration in sample_times and self.drop and len(self.particles) > 10:
                drop = True

            self.update_particles(drop)
            drop =  False
            # if iteration % 10 == 0:
                # print(f"Iteration {iteration}: Global Best Score = {self.global_best_score}")
            if self.has_converged():
                #print("Converged!")
                break
            self.prev_best_score = self.global_best_score



def rosenbrock_function(pos):
    x, y = pos
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + sum([xi**2 - A * np.cos(2 * np.pi * xi) for xi in x])




def create_animation(pso, bounds, interval=50, save_as_mp4=False, filename='pso_animation.mp4'):
    fig, ax = plt.subplots()
    x = np.linspace(bounds[0][0], bounds[0][1], 4000)
    y = np.linspace(bounds[1][0], bounds[1][1], 4000)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_function([X, Y])
    levels = np.linspace(0, 50, 50)
    X_levels, Y_levels = np.meshgrid(levels, levels)
    Z_levels = np.sort(rosenbrock_function([X_levels, Y_levels]))[0]

    contour = ax.contour(X, Y, Z, levels=Z_levels, colors='black')
    scat = ax.scatter(pso.particles[:, 0], pso.particles[:, 1], color='red')
    ax.scatter(*pso.global_min, color='blue', marker='x', s=100)  # Global minimum

    def update(frame):
        pso.update_particles()
        scat.set_offsets(pso.particles)
        print(f"Iteration {frame}: Global Best Score = {pso.global_best_score}")
        if pso.has_converged():
            print("Converged!")
            ani.event_source.stop()
        pso.prev_best_score = pso.global_best_score
        return scat,

    ani = FuncAnimation(fig, update, frames=range(pso.max_iter), interval=interval, blit=True)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Particle Swarm Optimization on Rosenbrock Function')

    if save_as_mp4:
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(filename, writer=writer)
    else:
        plt.show()

    return ani  # Ensure the animation object is returned to keep a reference


def main():
    bounds = [(-10, 10), (-10, 10)]
    global_min = [1, 1]  # True global minimum for the Rosenbrock function
    tolerance = 1e-3
    num_of_stable_iterations = 5
    initial_num_particles = 50
    w = 0.2
    c1 = 1.6
    c2 = 0.6
    pso = PSO(w=w , c1=c1, c2=c2, func=rosenbrock_function, bounds=bounds, global_min=global_min, tol=tolerance, stable_iterations=num_of_stable_iterations, initial_num_particles=initial_num_particles, drop=True)
    # print("Running PSO without animation...")
    # pso.run()
    print("Running PSO with animation...")
    create_animation(pso, bounds, interval=500, save_as_mp4=False)


if __name__ == "__main__":
    main()