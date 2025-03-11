import numpy as np

def ez_diffusion(a, v, t, num_trials=1): # Simulate the EZ diffusion model for a number of trials.
    # Parameters:
       # a (float): Boundary separation (decision boundary).
       # v (float): Drift rate (the rate of evidence accumulation).
       # t (float): Nondecision time (time before the decision process starts).
       # num_trials (int): Number of trials to simulate (default is 1).
    
    # Returns:
        # reaction_times (numpy array): Simulated reaction times for each trial.
        # accuracy (numpy array): Simulated accuracy for each trial (1 for correct, 0 for incorrect).
    
    reaction_times = []
    accuracy = []
    
    for _ in range(num_trials):
        # Simulate the decision process
        # Initialize the accumulated evidence
        evidence = 0
        time = 0
        while abs(evidence) < a:
            # Brownian motion with drift (v) and random noise (standard normal)
            evidence += v + np.random.normal(0, 1)
            time += 1
        
        # Total reaction time is nondecision time + the time it takes to reach a boundary
        reaction_time = t + time
        
        # Determine accuracy based on which boundary was hit
        if evidence >= a:
            # Correct decision (assuming drift rate is positive)
            accuracy.append(1)
        else:
            # Incorrect decision (assuming drift rate is negative)
            accuracy.append(0)
        
        # Record the reaction time for the trial
        reaction_times.append(reaction_time)
    
    return np.array(reaction_times), np.array(accuracy)

# Example usage
if __name__ == "__main__":
    # Define model parameters
    a = 1.0  # boundary separation
    v = 1.5  # drift rate
    t = 0.2  # nondecision time

    # Simulate 1000 trials
    reaction_times, accuracy = ez_diffusion(a, v, t, num_trials=1000)
    
    # Output the results
    print(f"Mean Reaction Time: {np.mean(reaction_times):.4f} seconds")
    print(f"Accuracy: {np.mean(accuracy):.4f}")
