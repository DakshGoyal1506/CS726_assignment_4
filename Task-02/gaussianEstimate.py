import numpy as np
import matplotlib.pyplot as plt

def branin_hoo(x):
    """Calculate the Branin-Hoo function value for given input.
        
        f(x1,x2) = (x2 - b*x1^2 + c*x1 - r)^2 + s*(1 - t)*cos(x1) + s
        with parameters:
         a = 1, b = 5.1/(4π²), c = 5/π, r = 6, s = 10, t = 1/(8π)
        Input:
            x: a list or numpy array with 2 elements: [x1, x2]
    """

    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    x1 = x[0]
    x2 = x[1]

    return (x2 - b * x1**2 + c*x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    

# Kernel Functions (Students implement)
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Compute the RBF kernel."""
    sqdist = np.sum((x1 - x2)**2)
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)
    pass

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Compute the Matérn kernel (nu=1.5)."""
    d = np.linalg.norm(x1 - x2)
    return sigma_f**2 * ( 1 + np.sqrt(3)*d / length_scale) * np.exp(-np.sqrt(3)*d / length_scale)
    pass

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Compute the Rational Quadratic kernel."""
    sqdist = np.sum((x1 - x2)**2)
    return sigma_f**2 * (1 + sqdist / (2 * alpha * length_scale**2))**(-alpha)
    pass

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
    """Compute the log-marginal likelihood."""
    n = x_train.shape[0]
    k = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            k[i, j] = kernel_func(x_train[i], x_train[j], length_scale, sigma_f)
    
    k = k + noise * np.eye(n)

    l = np.linalg.cholesky(k + 1e-6 * np.eye(n))
    alpha = np.linalg.solve(l.T, np.linalg.solve(l, y_train))

    log_likekihood = -0.5 * np.dot(y_train, alpha) - np.sum(np.log(np.diag(l))) - (n / 2) * np.log(2 * np.pi)

    return log_likekihood
    pass

def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """Optimize hyperparameters using grid search."""
    best_ll = -np.inf
    best_params = (1.0, 1.0, noise)

    length_f_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    sigma_f_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    noise_vals = [1e-4, 1e-3, 1e-2]

    for ls in length_f_vals:
        for sf in sigma_f_vals:
            for nv in noise_vals:
                ll = log_marginal_likelihood(x_train=x_train,
                                             y_train=y_train,
                                             kernel_func=kernel_func,
                                             length_scale=ls,
                                             sigma_f=sf,
                                             noise=nv)
                if(ll > best_ll):
                    best_ll = ll
                    best_params = (ls, sf, nv)
    
    return best_params
    pass

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    """Perform GP prediction."""
    n = x_train.shape[0]
    m = x_test.shape[0]

    k = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            k[i, j] = kernel_func(x_train[i], x_train[j], length_scale, sigma_f)
    
    k = k + noise * np.eye(n)

    l = np.linalg.cholesky(k + 1e-6 * np.eye(n))

    k_s = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            k_s[i, j] = kernel_func(x_train[i], x_test[j], length_scale, sigma_f)
    
    k_s_diagnol = np.array([kernel_func(x_test[j], x_test[j], length_scale, sigma_f) for j in range(m)]) + noise

    alpha = np.linalg.solve(l.T, np.linalg.solve(l, y_train))
    y_mean = k_s.T.dot(alpha)

    v = np.linalg.solve(l, k_s)

    y_var = k_s_diagnol - np.sum(v**2, axis=0)
    y_std = np.sqrt(np.maximum(y_var, 0))

    return y_mean, y_std
    pass


def Phi(z) :
    """Approximate standard normal CDF using a logistic function."""
    return (1 / (1 + np.exp(-1.702 * z)))

def D_Phi(z):
    """Approximate standard normal PDF, derivative of the above approximation."""
    return (1.702 * np.exp(-1.702 * z) / (1 + np.exp(-1.702 * z)) ** 2)

# Acquisition Functions (Simplified, no erf)
def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Expected Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    imporvement = mu - y_best - xi

    z = imporvement / (sigma + 1e-9)

    phi_z = Phi(z)
    d_phi_z = D_Phi(z)

    return (imporvement * phi_z + sigma * d_phi_z)
    pass

def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Probability of Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    improvement = mu - y_best - xi
    z = improvement / (sigma + 1e-9)

    return Phi(z)
    pass

def plot_graph(x1_grid, x2_grid, z_values, x_train, title, filename):
    """Create and save a contour plot."""

    plt.figure()
    cp = plt.contourf(x1_grid, x2_grid, z_values)
    plt.scatter(x_train[:, 0], x_train[:, 1], c='r', marker='x')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar(cp)
    plt.savefig(filename)
    plt.close()

def main():
    """Main function to run GP with kernels, sample sizes, and acquisition functions."""
    np.random.seed(0)
    n_samples_list = [10, 20, 50, 100]
    kernels = {
        'rbf': (rbf_kernel, 'RBF'),
        'matern': (matern_kernel, 'Matern (nu=1.5)'),
        'rational_quadratic': (rational_quadratic_kernel, 'Rational Quadratic')
    }
    acquisition_strategies = {
        'EI': expected_improvement,
        'PI': probability_of_improvement
    }
    
    x1_test = np.linspace(-5, 10, 100)
    x2_test = np.linspace(0, 15, 100)
    x1_grid, x2_grid = np.meshgrid(x1_test, x2_test)
    x_test = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    true_values = np.array([branin_hoo([x1, x2]) for x1, x2 in x_test]).reshape(x1_grid.shape)
    
    for kernel_name, (kernel_func, kernel_label) in kernels.items():
        for n_samples in n_samples_list:
            x_train = np.random.uniform(low=[-5, 0], high=[10, 15], size=(n_samples, 2))
            y_train = np.array([branin_hoo(x) for x in x_train])
            
            print(f"\nKernel: {kernel_label}, n_samples = {n_samples}")
            length_scale, sigma_f, noise = optimize_hyperparameters(x_train, y_train, kernel_func)
            
            for acq_name, acq_func in acquisition_strategies.items():
                x_train_current = x_train.copy()
                y_train_current = y_train.copy()
                
                y_mean, y_std = gaussian_process_predict(x_train_current, y_train_current, x_test, 
                                                        kernel_func, length_scale, sigma_f, noise)
                y_mean_grid = y_mean.reshape(x1_grid.shape)
                y_std_grid = y_std.reshape(x1_grid.shape)
                
                if acq_func is not None:
                    # Hint: Find y_best, apply acq_func, select new point, update training set, recompute GP
                    pass
                
                acq_label = '' if acq_name == 'None' else f', Acq={acq_name}'
                plot_graph(x1_grid, x2_grid, true_values, x_train_current,
                          f'True Branin-Hoo Function (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'true_function_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_mean_grid, x_train_current,
                          f'GP Predicted Mean (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_mean_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_std_grid, x_train_current,
                          f'GP Predicted Std Dev (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_std_{kernel_name}_n{n_samples}_{acq_name}.png')

if __name__ == "__main__":
    main()