"""
Method 1: Beta Distribution Parameterization
Theoretically correct: Model outputs Beta distribution parameters (α, β)
Score = α/(α+β), Uncertainty = variance of Beta distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, betaln
from scipy.stats import beta as beta_dist
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class BetaDistributionPU:
    """
    PU Learning via Beta Distribution Parameterization
    
    Model predicts (α, β) such that:
    - Score = α/(α+β) ∈ [0, 1]
    - Uncertainty = αβ/[(α+β)²(α+β+1)]
    """
    
    def __init__(self, alpha_prior=1.0, beta_prior=1.0, min_param=0.1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.min_param = min_param
        self.params = None
        
    def _softplus(self, x):
        """Softplus ensures positivity: log(1 + exp(x))"""
        return np.log1p(np.exp(np.clip(x, -20, 20)))
    
    def _predict_params(self, X, w_alpha, b_alpha, w_beta, b_beta):
        """Predict Beta distribution parameters"""
        # Linear combination
        z_alpha = X @ w_alpha + b_alpha
        z_beta = X @ w_beta + b_beta
        
        # Ensure positivity with softplus + minimum
        alpha = self._softplus(z_alpha) + self.min_param
        beta = self._softplus(z_beta) + self.min_param
        
        return alpha, beta
    
    def _beta_mean(self, alpha, beta):
        """Mean of Beta distribution"""
        return alpha / (alpha + beta)
    
    def _beta_variance(self, alpha, beta):
        """Variance of Beta distribution"""
        ab_sum = alpha + beta
        return (alpha * beta) / (ab_sum ** 2 * (ab_sum + 1))
    
    def _kl_divergence_beta(self, alpha, beta, alpha_prior, beta_prior):
        """KL divergence between Beta(α,β) and Beta(α₀,β₀)"""
        return (betaln(alpha_prior, beta_prior) - betaln(alpha, beta) +
                (alpha - alpha_prior) * digamma(alpha) +
                (beta - beta_prior) * digamma(beta) +
                (alpha_prior - alpha + beta_prior - beta) * digamma(alpha + beta))
    
    def _loss(self, X_pos, X_unl, w_alpha, b_alpha, w_beta, b_beta, 
              kl_weight=0.01, margin=0.1):
        """
        Loss for PU learning with Beta distributions
        
        For positive samples: maximize α, minimize β (push mean → 1)
        For unlabeled samples: minimize α, maximize β (push mean → 0)
        """
        # Predict parameters
        alpha_pos, beta_pos = self._predict_params(X_pos, w_alpha, b_alpha, w_beta, b_beta)
        alpha_unl, beta_unl = self._predict_params(X_unl, w_alpha, b_alpha, w_beta, b_beta)
        
        # Means
        mean_pos = self._beta_mean(alpha_pos, beta_pos)
        mean_unl = self._beta_mean(alpha_unl, beta_unl)
        
        # Ranking loss: positive means should be higher than unlabeled
        differences = mean_pos[:, np.newaxis] - mean_unl[np.newaxis, :]
        ranking_loss = np.maximum(0, margin - differences).mean()
        
        # KL divergence to prior (regularization)
        kl_pos = self._kl_divergence_beta(alpha_pos, beta_pos, 
                                          self.alpha_prior, self.beta_prior).mean()
        kl_unl = self._kl_divergence_beta(alpha_unl, beta_unl,
                                          self.alpha_prior, self.beta_prior).mean()
        
        kl_loss = kl_pos + kl_unl
        
        # Total loss
        total_loss = ranking_loss + kl_weight * kl_loss
        
        return total_loss, ranking_loss, kl_loss
    
    def fit(self, X_pos, X_unl, lr=0.05, n_iterations=1000, verbose=True):
        """Fit the Beta distribution model"""
        if X_pos.ndim == 1:
            X_pos = X_pos.reshape(-1, 1)
        if X_unl.ndim == 1:
            X_unl = X_unl.reshape(-1, 1)
            
        d = X_pos.shape[1]
        
        # Initialize parameters
        w_alpha = np.random.randn(d, 1) * 0.1
        b_alpha = np.random.randn() * 0.1
        w_beta = np.random.randn(d, 1) * 0.1
        b_beta = np.random.randn() * 0.1
        
        for iteration in range(n_iterations):
            # Forward pass
            alpha_pos, beta_pos = self._predict_params(X_pos, w_alpha, b_alpha, w_beta, b_beta)
            alpha_unl, beta_unl = self._predict_params(X_unl, w_alpha, b_alpha, w_beta, b_beta)
            
            mean_pos = self._beta_mean(alpha_pos, beta_pos)
            mean_unl = self._beta_mean(alpha_unl, beta_unl)
            
            # Compute loss
            loss, rank_loss, kl_loss = self._loss(X_pos, X_unl, w_alpha, b_alpha, 
                                                   w_beta, b_beta)
            
            # Gradients (numerical for simplicity and correctness)
            eps = 1e-5
            
            # Gradient w.r.t. w_alpha
            grad_w_alpha = np.zeros_like(w_alpha)
            for i in range(len(w_alpha)):
                w_alpha[i] += eps
                loss_plus, _, _ = self._loss(X_pos, X_unl, w_alpha, b_alpha, w_beta, b_beta)
                w_alpha[i] -= 2 * eps
                loss_minus, _, _ = self._loss(X_pos, X_unl, w_alpha, b_alpha, w_beta, b_beta)
                w_alpha[i] += eps
                grad_w_alpha[i] = (loss_plus - loss_minus) / (2 * eps)
            
            # Gradient w.r.t. b_alpha
            b_alpha += eps
            loss_plus, _, _ = self._loss(X_pos, X_unl, w_alpha, b_alpha, w_beta, b_beta)
            b_alpha -= 2 * eps
            loss_minus, _, _ = self._loss(X_pos, X_unl, w_alpha, b_alpha, w_beta, b_beta)
            b_alpha += eps
            grad_b_alpha = (loss_plus - loss_minus) / (2 * eps)
            
            # Gradient w.r.t. w_beta
            grad_w_beta = np.zeros_like(w_beta)
            for i in range(len(w_beta)):
                w_beta[i] += eps
                loss_plus, _, _ = self._loss(X_pos, X_unl, w_alpha, b_alpha, w_beta, b_beta)
                w_beta[i] -= 2 * eps
                loss_minus, _, _ = self._loss(X_pos, X_unl, w_alpha, b_alpha, w_beta, b_beta)
                w_beta[i] += eps
                grad_w_beta[i] = (loss_plus - loss_minus) / (2 * eps)
            
            # Gradient w.r.t. b_beta
            b_beta += eps
            loss_plus, _, _ = self._loss(X_pos, X_unl, w_alpha, b_alpha, w_beta, b_beta)
            b_beta -= 2 * eps
            loss_minus, _, _ = self._loss(X_pos, X_unl, w_alpha, b_alpha, w_beta, b_beta)
            b_beta += eps
            grad_b_beta = (loss_plus - loss_minus) / (2 * eps)
            
            # Update with gradient clipping
            w_alpha -= lr * np.clip(grad_w_alpha, -1, 1)
            b_alpha -= lr * np.clip(grad_b_alpha, -1, 1)
            w_beta -= lr * np.clip(grad_w_beta, -1, 1)
            b_beta -= lr * np.clip(grad_b_beta, -1, 1)
            
            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: Loss={loss:.4f} (Rank={rank_loss:.4f}, KL={kl_loss:.4f})")
        
        self.params = {
            'w_alpha': w_alpha,
            'b_alpha': b_alpha,
            'w_beta': w_beta,
            'b_beta': b_beta
        }
        
        return self
    
    def predict(self, X):
        """Predict Beta distribution parameters and statistics"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        alpha, beta = self._predict_params(X, 
                                           self.params['w_alpha'],
                                           self.params['b_alpha'],
                                           self.params['w_beta'],
                                           self.params['b_beta'])
        
        mean = self._beta_mean(alpha, beta)
        variance = self._beta_variance(alpha, beta)
        uncertainty = np.sqrt(variance)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'mean': mean,
            'variance': variance,
            'uncertainty': uncertainty
        }


def visualize_beta_method(X_pos, X_unl, model, option_name):
    """Visualize Beta distribution PU learning"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    x_range = np.linspace(X_unl.min() - 1, X_pos.max() + 1, 300).reshape(-1, 1)
    
    # Predictions
    pred_pos = model.predict(X_pos)
    pred_unl = model.predict(X_unl)
    pred_range = model.predict(x_range)
    
    # 1. Data distributions
    ax = axes[0, 0]
    ax.hist(X_pos.flatten(), bins=30, alpha=0.6, color='green', density=True, label='Positive')
    ax.hist(X_unl.flatten(), bins=30, alpha=0.6, color='red', density=True, label='Unlabeled')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Data Distributions ({option_name})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Beta parameters (α, β)
    ax = axes[0, 1]
    ax.plot(x_range, pred_range['alpha'], 'b-', label='α (alpha)', linewidth=2)
    ax.plot(x_range, pred_range['beta'], 'r-', label='β (beta)', linewidth=2)
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Beta Distribution Parameters', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, 'High α, low β → score→1\nLow α, high β → score→0\nα≈β → score≈0.5',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 3. Mean scores (α/(α+β))
    ax = axes[0, 2]
    ax.scatter(X_pos.flatten(), pred_pos['mean'], c='green', alpha=0.6, s=30, label='Positive')
    ax.scatter(X_unl.flatten(), pred_unl['mean'], c='red', alpha=0.6, s=30, label='Unlabeled')
    ax.plot(x_range, pred_range['mean'], 'k-', linewidth=2, label='Mean function')
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Score (Mean of Beta)')
    ax.set_title('Predicted Scores', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # 4. Uncertainty (std of Beta)
    ax = axes[1, 0]
    ax.plot(x_range, pred_range['uncertainty'], 'purple', linewidth=2)
    ax.fill_between(x_range.flatten(), pred_range['uncertainty'].flatten(), alpha=0.4, color='purple')
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Uncertainty (Std of Beta)')
    ax.set_title('Automatic Uncertainty Quantification', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.95, 'Uncertainty arises from\nBeta distribution variance',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 5. Score with uncertainty bands
    ax = axes[1, 1]
    mean = pred_range['mean'].flatten()
    unc = pred_range['uncertainty'].flatten()
    ax.plot(x_range, mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(x_range.flatten(), mean - unc, mean + unc, alpha=0.3, color='blue', label='±1 std')
    ax.fill_between(x_range.flatten(), mean - 2*unc, mean + 2*unc, alpha=0.15, color='blue', label='±2 std')
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Score')
    ax.set_title('Score with Uncertainty Bands', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # 6. Beta distributions at specific points
    ax = axes[1, 2]
    
    # Select 3 representative points
    idx_low = len(x_range) // 4
    idx_mid = len(x_range) // 2
    idx_high = 3 * len(x_range) // 4
    
    x_vals = np.linspace(0, 1, 100)
    for idx, color, label in [(idx_low, 'red', 'Low value'),
                               (idx_mid, 'orange', 'Middle'),
                               (idx_high, 'green', 'High value')]:
        alpha_val = float(pred_range['alpha'][idx])
        beta_val = float(pred_range['beta'][idx])
        pdf = beta_dist.pdf(x_vals, alpha_val, beta_val)
        ax.plot(x_vals, pdf, color=color, linewidth=2, 
                label=f'{label}: α={alpha_val:.2f}, β={beta_val:.2f}')
    
    ax.set_xlabel('Score Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('Beta Distributions at Different Inputs', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_data(option='medium', n_pos=100, n_unl=500):
    """Generate test data"""
    if option == 'easy':
        X_pos = np.random.normal(-1.0, 0.5, n_pos)
        X_unl = np.random.normal(-3.0, 0.8, n_unl)
    elif option == 'medium':
        X_pos = np.random.normal(-1.5, 1.0, n_pos)
        X_unl = np.random.normal(-2.5, 1.2, n_unl)
    elif option == 'hard':
        X_pos = np.random.normal(-2.0, 1.5, n_pos)
        X_unl = np.random.normal(-2.5, 1.5, n_unl)
    return X_pos.reshape(-1, 1), X_unl.reshape(-1, 1)


if __name__ == "__main__":
    print("="*70)
    print("METHOD 1: Beta Distribution Parameterization")
    print("="*70)
    
    for option in ['easy', 'medium', 'hard']:
        print(f"\n--- Testing {option.upper()} distribution ---")
        
        X_pos, X_unl = generate_data(option)
        
        model = BetaDistributionPU(alpha_prior=1.0, beta_prior=1.0, min_param=0.1)
        model.fit(X_pos, X_unl, lr=0.01, n_iterations=500, verbose=False)
        
        # Evaluate
        pred_pos = model.predict(X_pos)
        pred_unl = model.predict(X_unl)
        
        print(f"Positive: mean={pred_pos['mean'].mean():.3f}, unc={pred_pos['uncertainty'].mean():.3f}")
        print(f"Unlabeled: mean={pred_unl['mean'].mean():.3f}, unc={pred_unl['uncertainty'].mean():.3f}")
        print(f"Separation: {pred_pos['mean'].mean() - pred_unl['mean'].mean():.3f}")
        
        fig = visualize_beta_method(X_pos, X_unl, model, option)
        plt.savefig(f'method1_beta_{option}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: method1_beta_{option}.png")
        plt.close()
    
    print("\n" + "="*70)
    print("Beta Distribution Method: COMPLETE")
    print("Theoretical Properties:")
    print("  ✓ Natural bounds [0,1]")
    print("  ✓ Automatic uncertainty from variance")
    print("  ✓ Non-zero scores (min_param > 0)")
    print("  ✓ α≈β → score≈0.5 (high entropy)")
    print("="*70)