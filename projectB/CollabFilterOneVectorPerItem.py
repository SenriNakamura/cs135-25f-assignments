'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np
import numpy as np
import matplotlib.pyplot as plt

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets


class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        random_state = self.random_state

        # Global mean of training ratings
        y_train = train_tuple[2]
        mu = ag_np.array([ag_np.mean(y_train)])

        # Biases
        b_per_user = ag_np.zeros(n_users)
        c_per_item = ag_np.zeros(n_items)

        # Latent factors
        U = 0.001 * random_state.randn(n_users, self.n_factors)
        V = 0.001 * random_state.randn(n_items, self.n_factors)

        self.param_dict = dict(
            mu=mu,
            b_per_user=b_per_user,
            c_per_item=c_per_item,
            U=U,
            V=V,
    )


    def predict(self, user_id_N, item_id_N,
        mu=None, b_per_user=None, c_per_item=None, U=None, V=None):

    # If params not provided, pull from self.param_dict
        if mu is None:
            mu = self.param_dict['mu']
        if b_per_user is None:
            b_per_user = self.param_dict['b_per_user']
        if c_per_item is None:
            c_per_item = self.param_dict['c_per_item']
        if U is None:
            U = self.param_dict['U']
        if V is None:
            V = self.param_dict['V']

        # Look up latent vectors
        u_vecs = U[user_id_N]      # (N, n_factors)
        v_vecs = V[item_id_N]      # (N, n_factors)

        # Interaction term
        interaction_N = ag_np.sum(u_vecs * v_vecs, axis=1)

        # Final prediction
        yhat_N = mu[0] + b_per_user[user_id_N] + c_per_item[item_id_N] + interaction_N
        return yhat_N

    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        user_id_N, item_id_N, y_N = data_tuple

        yhat_N = self.predict(user_id_N, item_id_N, **param_dict)

        err_N = yhat_N - y_N
        data_loss = 0.5 * ag_np.mean(err_N ** 2)

        b_per_user = param_dict['b_per_user']
        c_per_item = param_dict['c_per_item']
        U = param_dict['U']
        V = param_dict['V']

        reg_loss = 0.5 * self.alpha * (
            ag_np.sum(b_per_user ** 2)
            + ag_np.sum(c_per_item ** 2)
            + ag_np.sum(U ** 2)
            + ag_np.sum(V ** 2)
        )

        return data_loss + reg_loss


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()

    # -------- Settings shared with 1a and 1b --------
    batch_size = 1000
    step_size = 5
    n_epochs = 10
    random_state = 0

    # ==========================================================
    # 1a: Train K = 2,10,50 with alpha = 0 and make 3-panel figure
    # ==========================================================
    alpha_1a = 0.0
    k_values = [2, 10, 50]
    histories = {}

    for k in k_values:
        print("\n" + "=" * 60)
        print(f"Training CollabFilterOneVectorPerItem with n_factors = {k}")
        print("=" * 60)

        model = CollabFilterOneVectorPerItem(
            n_epochs=n_epochs,
            batch_size=batch_size,
            step_size=step_size,
            n_factors=k,
            alpha=alpha_1a,
            random_state=random_state,
        )
        model.init_parameter_dict(n_users, n_items, train_tuple)
        model.fit(train_tuple, valid_tuple)

        # Store traces
        histories[k] = dict(
            epoch=np.array(model.trace_epoch),
            tr_rmse=np.array(model.trace_rmse_train),
            va_rmse=np.array(model.trace_rmse_valid),
        )

    # ---------- Make Figure 1a: 3 panels side-by-side ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, k in zip(axes, k_values):
        h = histories[k]
        ax.plot(h['epoch'], h['tr_rmse'], label='Train RMSE')
        ax.plot(h['epoch'], h['va_rmse'], linestyle='--', label='Valid RMSE')
        ax.set_title(f'{k} factors')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('RMSE')
    axes[0].legend(loc='upper right')

    fig.suptitle('RMSE vs. Epoch for CollabFilterOneVectorPerItem (λ = 0)')
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # 1b: Grid search over alpha for K = 50 and make single-panel figure
    # ==========================================================
    alpha_grid = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
    k_1b = 50

    best_alpha = None
    best_val_rmse = np.inf
    best_model = None

    for a in alpha_grid:
        print("\n" + "=" * 60)
        print(f"Grid search: K = {k_1b}, alpha = {a}")
        print("=" * 60)

        model = CollabFilterOneVectorPerItem(
            n_epochs=n_epochs,
            batch_size=batch_size,
            step_size=step_size,   # if you tune step_size, change here
            n_factors=k_1b,
            alpha=a,
            random_state=random_state,
        )
        model.init_parameter_dict(n_users, n_items, train_tuple)
        model.fit(train_tuple, valid_tuple)

        final_val_rmse = model.trace_rmse_valid[-1]
        print(f"Final valid RMSE for alpha={a}: {final_val_rmse:.5f}")

        if final_val_rmse < best_val_rmse:
            best_val_rmse = final_val_rmse
            best_alpha = a
            best_model = model

    print("\nBest alpha:", best_alpha)
    print("Best validation RMSE:", best_val_rmse)

    # ---------- Make Figure 1b: single panel for best alpha ----------
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(best_model.trace_epoch,
            best_model.trace_rmse_train,
            label='Train RMSE')
    ax.plot(best_model.trace_epoch,
            best_model.trace_rmse_valid,
            linestyle='--',
            label='Valid RMSE')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title(f'RMSE vs. Epoch for K = {k_1b}, α = {best_alpha:g}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
