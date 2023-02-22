import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    n = x.shape[0]
    split_idx = np.random.choice(K, size = n, replace = True)
    mu = np.zeros((K, x.shape[1]))
    sigma = np.zeros((K, x.shape[1], x.shape[1]))
    mu = [np.mean(x[split_idx == y, :], axis = 0) for y in range(K)]
    sigma = [np.cov(x[split_idx == y, :].T) for y in range(K)]
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones((K)) / K
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones((x.shape[0], K)) / K
    # print(phi)
    # print(mu)
    # print(sigma)
    # print(w)
    # *** END CODE HERE ***
    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)
    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # for i in range(x.shape[0]):
        #     for j in range(K):
        #         sigma_inv = np.linalg.inv(sigma[j])
        #         constant_part = 1./((np.power(2. * np.pi, x.shape[1] / 2)) * (np.sqrt(np.linalg.det(sigma[j]))))
        #         x_mu = np.reshape(x[i] - mu[j], (x.shape[1], 1))
        #         exp_part = np.exp((-1/2) * (x_mu.T) @ sigma_inv @ (x_mu))
        #         # for l in range(K):
        #         #     sigma_inv_1 = np.linalg.pinv(sigma[l])
        #         #     constant_part_1 = (1 / (np.power(2 * np.pi, x.shape[1] / 2)) * (np.sqrt(np.linalg.det(sigma[l]))))
        #         #     exp_part_1 = np.exp((-1 / 2) * (x[i] - mu[l]) @ sigma_inv_1 @ (x[i] - mu[l]).T)
        #         #     constant_l += constant_part_1 * exp_part_1 * phi[l]
        #         w[i, j] = constant_part * exp_part * phi[j]
        # w /= np.sum(w, axis = 1, keepdims =True)
        #print(w)
        w = find_w(phi, mu, sigma, x)
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # phi = np.mean(w, axis = 0)
        # sigma = np.zeros((K, x.shape[1], x.shape[1]))
        # for i in range(K):
        #     mu[i] = w[:,i].T @ x / (np.sum(w[:,i]))
        #     for y in range(x.shape[0]):
        #         x_mu = np.reshape(x[y] - mu[i], (x.shape[1], 1))
        #         sigma[i] += w[y,i] * (x_mu) @ (x_mu).T
        #         #print(sigma[i])
        #     sigma[i] = sigma[i] / np.sum(w[:,i])
        phi = find_phi_unsuper(w)
        mu = find_mu_unsuper(w, x)
        sigma = find_sigma_unsuper(w,x,mu)
        # (3) Compute the log-likelihood of the data to check for convergence.
        #print(sigma[1])
        #print(mu[1])
        #print(phi[1])
        prev_ll = ll
        ll = 0
        for i in range(x.shape[0]):
            ll_c = 0
            for j in range(K):
                sigma_inv = np.linalg.inv(sigma[j])
                constant_part = (1/((np.power(2 * np.pi, x.shape[1] / 2)) * (np.sqrt(np.linalg.det(sigma[j])))))
                x_mu = np.reshape(x[i] - mu[j], (x.shape[1], 1))
                exp_part = np.exp((-1/2) * (x_mu.T) @ sigma_inv @ (x_mu))
                ll_c += constant_part * exp_part * phi[j]
                #denominator = w[i,j]
            ll += np.log(ll_c)
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        #print(ll)
        it += 1
        print(it)
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        w = find_w(phi, mu, sigma, x)
        # for i in range(x.shape[0]):
        #     for j in range(K):
        #         sigma_inv = np.linalg.inv(sigma[j])
        #         constant_part = 1. / ((np.power(2. * np.pi, x.shape[1] / 2)) * (np.sqrt(np.linalg.det(sigma[j]))))
        #         x_mu = np.reshape(x[i] - mu[j], (x.shape[1], 1))
        #         exp_part = np.exp((-1 / 2) * (x_mu.T) @ sigma_inv @ (x_mu))
        #         w[i, j] = constant_part * exp_part * phi[j]
        # w /= np.sum(w, axis=1, keepdims=True)
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # phi
        phi = find_phi_semisuper(w, z_tilde, alpha)
        # for j in range(K):
        #     numerator = np.sum(w[:,j]) + alpha * np.sum(z_tilde == j)
        #     denominator = x.shape[0] + alpha * x_tilde.shape[0]
        #     phi[j] = numerator / denominator
        # # mu
        mu = find_mu_semisuper(w, x, x_tilde, z_tilde, alpha)
        # for j in range(K):
        #     z_tilde = z_tilde.reshape(z_tilde.shape[0],)
        #     numerator = (w[:,j].T @ x) + alpha * np.sum(x_tilde[(z_tilde == j),:], axis = 0)
        #     denominator = np.sum(w[:, j]) + alpha * np.sum(z_tilde ==j)
        #     mu[j] = numerator / denominator
        #print(mu[1])
        # sigma
        # sigma = np.zeros((K, x.shape[1], x.shape[1]))
        # for j in range(K):
        #     # numerator
        #     for i in range(x.shape[0]):
        #         x_mu = np.reshape(x[i] - mu[j], (x.shape[1], 1))
        #         sigma[j] += w[i,j] * (x_mu) @ (x_mu).T
        #     for i in range(x_tilde.shape[0]):
        #         if z_tilde[i] == j:
        #             x_mu = np.reshape(x[i] - mu[j], (x_tilde.shape[1], 1))
        #             sigma[j] += alpha * (x_mu) @ (x_mu).T
        #     # denominator
        #     denominator = np.sum(w[:,j]) + alpha * np.sum(z_tilde ==j)
        #     sigma[j] /= denominator
        sigma = find_sigma_semisuper(w, x, mu, x_tilde, z_tilde, alpha)

        #print(sigma[1])
        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll = 0
        ll_super = 0
        for i in range(x.shape[0]):
            ll_c = 0
            for j in range(K):
                sigma_inv = np.linalg.inv(sigma[j])
                constant_part = (1/((np.power(2 * np.pi, x.shape[1] / 2)) * (np.sqrt(np.linalg.det(sigma[j])))))
                x_mu = np.reshape(x[i] - mu[j], (x.shape[1], 1))
                exp_part = np.exp((-1/2) * (x_mu.T) @ sigma_inv @ (x_mu))
                ll_c += constant_part * exp_part * phi[j]
                #denominator = w[i,j]
            ll += np.log(ll_c)
        for i in range(x_tilde.shape[0]):
            j = int(z_tilde[i])
            sigma_inv = np.linalg.inv(sigma[j])
            constant_part = (1 / ((np.power(2 * np.pi, x.shape[1] / 2)) * (np.sqrt(np.linalg.det(sigma[j])))))
            x_mu = np.reshape(x[i] - mu[j], (x.shape[1], 1))
            exp_part = np.exp((-1 / 2) * (x_mu.T) @ sigma_inv @ (x_mu))
            ll_c = constant_part * exp_part * phi[j]
            ll_super += np.log(ll_c)
        ll += alpha * ll_super
        #print(ll)
        it += 1
        print(it)
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***


    return w


# *** START CODE HERE ***
# Helper functions
def find_w(phi, mu, sigma, x):
    w = np.zeros((x.shape[0], K))
    for i in range(K):
        sigma_inv = np.linalg.inv(sigma[i])
        constant_part = 1. / ((np.power(2. * np.pi, x.shape[1] / 2)) * (np.sqrt(np.linalg.det(sigma[i]))))
        for y in range(x.shape[0]):
            #print(x[y].shape)
            #print(mu[i].shape)
            x_mu = (x[y] - mu[i]).reshape(2, 1)
            exp_part = np.exp((-1/2) * x_mu.T @ sigma_inv @ x_mu)
            w[y,i] = constant_part * exp_part * phi[i]
    w /= np.sum(w, axis = 1, keepdims = True)
    #print(w[1])
    return w

def find_phi_unsuper(w):
    phi = np.mean(w, axis = 0)
    return phi

def find_phi_semisuper(w, z, alpha):
    phi = np.zeros(4)
    for j in range(K):
        unsuper = np.sum(w[:,j])
        super = np.sum(z == j) * alpha
        phi[j] = unsuper + super

    return phi / (w.shape[0] + z.shape[0] * alpha)

def find_mu_unsuper(w, x):
    mu = np.zeros((K, x.shape[1]))
    for j in range(K):
        unsuper = w[:, j].T @ x
        mu[j] = unsuper
        denominator = np.sum(w[:,j])
        mu[j] /= denominator
    return mu


def find_mu_semisuper(w, x, x_hat, z, alpha):
    mu = np.zeros((K, x.shape[1]))
    for j in range(K):
        # unsuper = w[:, j].T @ x
        # z_1 = np.reshape(z, (z.shape[0],1))
        # z_1 = np.append(z_1,z_1,1)
        # super = np.sum(x_hat[z_1 == j], axis = 0) * alpha
        # mu[j] = unsuper + super
        # denominator = np.sum(w[:,j]) + np.sum(z == j) * alpha
        # mu[j] /= denominator
        super = 0
        unsuper = 0
        for i in range(x.shape[0]):
            super += w[i,j] * x[i]
        for i in range(x_hat.shape[0]):
            if z[i] == j:
                unsuper += x_hat[i] * alpha
        mu[j] = super + unsuper
        #print('super', super, 'unsuper',unsuper,mu[j])
        mu[j] /= (np.sum(w[:,j]) + np.sum(z == j) * alpha)
    return mu


def find_sigma_unsuper(w, x, mu):
    sigma = np.zeros((K, x.shape[1], x.shape[1]))
    for j in range(K):
        unsuper = 0
        for i in range(x.shape[0]):
            x_mu = (x[i] - mu[j]).reshape(2,1)
            unsuper += w[i,j] * x_mu @ x_mu.T
        sigma[j] += unsuper
        sigma[j] /= np.sum(w[:,j])
    return sigma


def find_sigma_semisuper(w, x, mu, x_hat, z, alpha):
    sigma = np.zeros((K, x.shape[1], x.shape[1]))
    for j in range(K):
        unsuper = 0
        super = 0
        for i in range(x.shape[0]):
            x_mu = (x[i] - mu[j]).reshape(2,1)
            unsuper += w[i,j] * x_mu @ x_mu.T
        for i in range(x_hat.shape[0]):
            if z[i] == j:
                x_mu_s = (x_hat[i] - mu[j]).reshape(2,1)
                super += (x_mu_s @ x_mu_s.T) * alpha
        sigma[j] += unsuper + super
        sigma[j] /= (np.sum(w[:,j]) + np.sum(z == j) * alpha)
    return sigma
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
