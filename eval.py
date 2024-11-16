import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from maml import MAMLRewardNetwork

# Load the population data
with open("maml_1000gen_pop.pkl", "rb") as f:
    data = pickle.load(f)

population = data['population']
fitness_scores = data['fitness_scores']

# Extract learning rates and convert to actual rates
learning_rates = []
for network in population:
    lr = 1 / (1 + np.exp(-network.lr_param)) * 0.1  # Sigmoid scaling
    learning_rates.append(lr)

learning_rates = np.array(learning_rates)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(learning_rates, fitness_scores, alpha=0.5)

# Add trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(learning_rates, fitness_scores)
line_x = np.array([min(learning_rates), max(learning_rates)])
line_y = slope * line_x + intercept
plt.plot(line_x, line_y, 'r--', label=f'R² = {r_value**2:.3f}')

plt.xlabel('Learning Rate')
plt.ylabel('Fitness Score')
plt.title('MAML Network Performance vs Learning Rate')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot
plt.savefig('learning_rate_analysis.png')

# Print summary statistics
print(f"Learning Rate Statistics:")
print(f"Mean: {np.mean(learning_rates):.6f}")
print(f"Std:  {np.std(learning_rates):.6f}")
print(f"Min:  {np.min(learning_rates):.6f}")
print(f"Max:  {np.max(learning_rates):.6f}")
print(f"\nCorrelation coefficient: {r_value:.3f}")
print(f"P-value: {p_value:.3e}")
