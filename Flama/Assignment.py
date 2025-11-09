import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Files\Edu\Python\Flama\xy_data.csv')
x_data = data['x'].values
y_data = data['y'].values

# Parametric equations
def parametric_x(t, theta, M, X):
    return t * np.cos(theta) - np.exp(M * t) * np.sin(0.3 * t) * np.sin(theta) + X

def parametric_y(t, theta, M):
    return 42 + t * np.sin(theta) + np.exp(M * t) * np.sin(0.3 * t) * np.cos(theta)


def cost_function(params, x_data, y_data):
    theta, M, X, t_min, t_max = params
    
    
    if t_max <= t_min:
        return 1e10 
    
    
    t_dense = np.linspace(t_min, t_max, 800)
    x_curve_dense = parametric_x(t_dense, theta, M, X)
    y_curve_dense = parametric_y(t_dense, theta, M)  
    
   
    x_diff = x_data[:, np.newaxis] - x_curve_dense[np.newaxis, :]
    y_diff = y_data[:, np.newaxis] - y_curve_dense[np.newaxis, :]
    
    # Calculate all squared distances
    distances_sq = x_diff**2 + y_diff**2
    
    # Find minimum distance for each data point
    min_distances_sq = np.min(distances_sq, axis=1)
    
    return np.sum(min_distances_sq)


initial_guess = [np.radians(15), 0.01, 60, 6, 60]  

bounds = [
    (np.radians(5), np.radians(60)),  
    (-0.1, 0.1),                      
    (0, 120),                         
    (0, 30),                          
    (30, 100)                         
]

# Optimize with multiple starting points to avoid local minima
print("Starting optimization with multiple initial guesses...")
best_result = None
best_cost = np.inf

# Try multiple initial guesses
initial_guesses = [
    [np.radians(15), 0.01, 60, 6, 60],   
    [np.radians(20), 0.02, 70, 5, 65],   
    [np.radians(10), -0.01, 50, 8, 55], 
    [np.radians(30), 0.03, 80, 4, 70],   
    [np.radians(25), 0.015, 65, 7, 58], 
    [np.radians(12), 0.008, 55, 10, 50],
]

for i, guess in enumerate(initial_guesses):
    print(f"  Trying initial guess {i+1}/{len(initial_guesses)}: theta={np.degrees(guess[0]):.1f} deg, M={guess[1]:.4f}, X={guess[2]:.1f}, t=[{guess[3]:.1f}, {guess[4]:.1f}]")
    result = minimize(
        cost_function,
        guess,
        args=(x_data, y_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-6}
    )
    if result.fun < best_cost:
        best_cost = result.fun
        best_result = result
        print(f"    -> New best cost: {best_cost:.2f}")

result = best_result
print(f"\nBest optimization result found with cost: {result.fun:.2f}")

# Extract results
theta_opt, M_opt, X_opt, t_min_opt, t_max_opt = result.x
theta_deg = np.degrees(theta_opt)

print("\n" + "="*50)
print("OPTIMIZED PARAMETERS:")
print("="*50)
print(f"theta = {theta_deg:.6f} degrees = {theta_opt:.6f} radians")
print(f"M = {M_opt:.6f}")
print(f"X = {X_opt:.6f}")
print(f"t_min = {t_min_opt:.6f}")
print(f"t_max = {t_max_opt:.6f}")
print(f"\nFinal cost (sum of squared distances): {result.fun:.6f}")
print("="*50)

# Generate fitted curve for plotting
t_plot = np.linspace(t_min_opt, t_max_opt, 500)
x_fit = parametric_x(t_plot, theta_opt, M_opt, X_opt)
y_fit = parametric_y(t_plot, theta_opt, M_opt)

# Create visualization
plt.figure(figsize=(12, 8))

# Plot original data points
plt.scatter(x_data, y_data, c='blue', s=10, alpha=0.5, label='Data points')

# Plot fitted curve
plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fitted parametric curve')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Parametric Curve Fitting', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')


textstr = f'theta = {theta_deg:.4f} deg\nM = {M_opt:.6f}\nX = {X_opt:.4f}\nt = [{t_min_opt:.2f}, {t_max_opt:.2f}]'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('parametric_fit.png', dpi=300, bbox_inches='tight')
plt.show()


print("\nFitted Parametric Equations:")
print(f"x = (t * cos({theta_deg:.4f}°) - e^({M_opt:.6f}*t) * sin(0.3t) * sin({theta_deg:.4f}°) + {X_opt:.4f})")
print(f"y = (42 + t * sin({theta_deg:.4f}°) + e^({M_opt:.6f}*t) * sin(0.3t) * cos({theta_deg:.4f}°))")
print(f"\nfor {t_min_opt:.2f} < t < {t_max_opt:.2f}")