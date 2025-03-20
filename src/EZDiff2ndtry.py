import numpy as np
import scipy.stats as stats

# Define parameter ranges
a_range = (0.5, 2)
v_range = (0.5, 2)
t_range = (0.1, 0.5)

# Sample sizes
N = [10, 40, 4000]
iterations = 1000

def generate_parameters():
    """Randomly generate true parameters."""
    a = np.random.uniform(*a_range)
    v = np.random.uniform(*v_range)
    t = np.random.uniform(*t_range)
    return a, v, t

def forward_equations(a, v, t):
    """Compute predicted summary statistics using forward equations."""
    y = np.exp(-a * v)
    R_pred = 1 / (y + 1)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((y + 1) ** 2))
    return R_pred, M_pred, V_pred

def simulate_observed_data(R_pred, M_pred, V_pred, N):
    """Simulate observed data based on predicted statistics."""
    T_obs = np.random.binomial(N, R_pred) / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    V_obs = stats.gamma.rvs((N - 1) / 2, scale=(2 * V_pred / (N - 1)))
    return T_obs, M_obs, V_obs

def inverse_equations(R_obs, M_obs, V_obs):
    """Recover estimated parameters using inverse equations."""
    epsilon = 1e-10 # Added epsilon from ChatGPT to avoid zero division error
    R_obs = np.clip(R_obs, epsilon, 1 - epsilon)
    L = np.log(R_obs / (1 - R_obs))
    v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(L * ((R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs))
    a_est = L / v_est
    t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
    return a_est, v_est, t_est

def compute_errors(true_params, est_params):
    """Compute bias and squared error."""
    bias = np.array(true_params) - np.array(est_params)
    squared_error = bias ** 2
    return bias, squared_error

def simulate_and_recover():
    """Run the full simulate-and-recover procedure."""
    results = []
    for N in [10, 40, 4000]:
        biases, errors = [], []
        for i in range(iterations):
            true_params = generate_parameters()
            R_pred, M_pred, V_pred = forward_equations(*true_params)
            R_obs, M_obs, V_obs = simulate_observed_data(R_pred, M_pred, V_pred, N)
            est_params = inverse_equations(R_obs, M_obs, V_obs)
            bias, squared_error = compute_errors(true_params, est_params)
            biases.append(bias)
            errors.append(squared_error)
        
        avg_bias = np.mean(biases, axis=0)
        avg_sq_error = np.mean(errors, axis=0)
        results.append((N, avg_bias, avg_sq_error))
    
    return results

if __name__ == "__main__":
    results = simulate_and_recover()
    for N, bias, error in results:
        print(f"Sample Size: {N}")
        print(f"Average Bias: {bias}")
        print(f"Average Squared Error: {error}\n")
