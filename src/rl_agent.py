import numpy as np

class LinUCB:
    def __init__(self, n_actions, d, alpha=1.0):
        """
        n_actions: number of arms (categories)
        d: dimension of context vector
        alpha: exploration parameter
        """
        self.n_actions = n_actions
        self.d = d
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_actions)]  # dxd identity matrices
        self.b = [np.zeros(d) for _ in range(n_actions)] # d-dim zero vectors

    def select(self, context):
        """
        Select action using LinUCB rule
        """
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
        return np.argmax(p)

    def update(self, a, context, r):
        """
        Update LinUCB parameters after observing reward
        a: action taken
        context: context vector
        r: observed reward
        """
        context = context.reshape(-1, 1)
        self.A[a] += context @ context.T
        self.b[a] += r * context.flatten()

    @property
    def thetas(self):
        """
        Compute theta vectors for all actions
        """
        return [np.linalg.inv(self.A[a]) @ self.b[a] for a in range(self.n_actions)]