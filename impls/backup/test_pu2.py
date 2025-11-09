# PU learning with elegant joint optimization of eta_p and tau
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ------------------------
# 1) Synthetic 1-D dataset
# ------------------------
n_pos = 600
n_neg = 1400
pos = rng.normal(loc=2.0, scale=0.7, size=n_pos)
neg = rng.normal(loc=.7, scale=1.7, size=n_neg)

pi_true = 0.35
n_u = 1600
n_u_pos = int(pi_true * n_u)
n_u_neg = n_u - n_u_pos
U = np.concatenate([
    rng.choice(pos, size=n_u_pos, replace=False),
    rng.choice(neg, size=n_u_neg, replace=False)
])
rng.shuffle(U)

rng.shuffle(pos)
split = int(0.6 * len(pos))
P_train = pos[:split]
P_val = pos[split:]

# ---------------------------------
# 2) Elegant joint optimization
# ---------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def train_elegant_pu(P_train, P_val, U, budget=0.05,
                     lr=1e-2, epochs=30000, l2=1e-4, temp=0.05,
                     lambda_fn=10.0, lambda_fp=100.0, seed=0):
    """
    Elegant joint optimization of (w, b, eta_p, tau).
    
    Objective:
        L_PU(w,b; eta_p) + lambda_fn * FN_soft(tau) + lambda_fp * [FP_soft(tau, eta_p) - budget]_+^2
    
    All variables optimized jointly via gradient descent.
    """
    rng_local = np.random.default_rng(seed)
    
    # Initialize parameters
    w = rng_local.normal(scale=0.1)
    b = 0.0
    eta_logit = 0.0  # eta_p = sigmoid(eta_logit), starts at 0.5
    tau_logit = 0.0  # tau = sigmoid(tau_logit), starts at 0.5
    
    for epoch in range(epochs):
        # Convert logits to probabilities
        eta_p = sigmoid(eta_logit)
        tau = sigmoid(tau_logit)
        
        # Forward pass
        yp_train = sigmoid(w * P_train + b)
        yp_val = sigmoid(w * P_val + b)
        yu = sigmoid(w * U + b)
        
        # ============================================
        # (1) PU Risk
        # ============================================
        L_pos = eta_p * (-np.mean(np.log(np.clip(yp_train, 1e-12, 1.0))))
        L_neg_raw = (np.mean(-np.log(np.clip(1 - yu, 1e-12, 1.0))) - 
                     eta_p * np.mean(-np.log(np.clip(1 - yp_train, 1e-12, 1.0))))
        L_neg = max(L_neg_raw, 0.0)
        L_reg = 0.5 * l2 * (w**2 + b**2)
        L_pu = L_pos + L_neg + L_reg
        
        # ============================================
        # (2) Soft False Negative (on validation)
        # ============================================
        # FN_soft = E[sigma((tau - y_val) / temp)]
        fn_soft = np.mean(sigmoid((tau - yp_val) / temp))
        
        # ============================================
        # (3) Soft False Positive (PU-corrected on mixture)
        # ============================================
        # FP_soft = E_U[sigma((y - tau)/temp)] - eta_p * E_P[sigma((y - tau)/temp)]
        surv_u = np.mean(sigmoid((yu - tau) / temp))
        surv_p = np.mean(sigmoid((yp_train - tau) / temp))
        fp_soft = surv_u - eta_p * surv_p
        
        # FP constraint penalty: [FP_soft - budget]_+^2
        fp_violation = max(0.0, fp_soft - budget)
        L_fp_penalty = lambda_fp * (fp_violation ** 2)
        
        # ============================================
        # Total Loss
        # ============================================
        L_total = L_pu + lambda_fn * fn_soft + L_fp_penalty
        
        # ============================================
        # Gradients w.r.t. (w, b)
        # ============================================
        # From PU loss
        dL_pos_dw = eta_p * np.mean((yp_train - 1) * P_train)
        dL_pos_db = eta_p * np.mean(yp_train - 1)
        
        if L_neg_raw > 0:
            dL_neg_dw = np.mean(yu * U) - eta_p * np.mean(yp_train * P_train)
            dL_neg_db = np.mean(yu) - eta_p * np.mean(yp_train)
        else:
            dL_neg_dw = 0.0
            dL_neg_db = 0.0
        
        dL_reg_dw = l2 * w
        dL_reg_db = l2 * b
        
        # From FN loss: d[sigma((tau - y)/temp)]/dy = -sigma'((tau-y)/temp)/temp
        fn_vals = sigmoid((tau - yp_val) / temp)
        d_fn_dyp = -(fn_vals * (1 - fn_vals)) / temp
        dL_fn_dw = lambda_fn * np.mean(d_fn_dyp * yp_val * (1 - yp_val) * P_val)
        dL_fn_db = lambda_fn * np.mean(d_fn_dyp * yp_val * (1 - yp_val))
        
        # From FP penalty: d[sigma((y - tau)/temp)]/dy = sigma'((y-tau)/temp)/temp
        if fp_violation > 0:
            surv_u_vals = sigmoid((yu - tau) / temp)
            surv_p_vals = sigmoid((yp_train - tau) / temp)
            
            d_surv_u_dyu = (surv_u_vals * (1 - surv_u_vals)) / temp
            d_surv_p_dyp = (surv_p_vals * (1 - surv_p_vals)) / temp
            
            # d(FP_penalty)/dy = 2 * lambda_fp * fp_violation * d(fp_soft)/dy
            dL_fp_dw = (2 * lambda_fp * fp_violation * 
                       (np.mean(d_surv_u_dyu * yu * (1 - yu) * U) - 
                        eta_p * np.mean(d_surv_p_dyp * yp_train * (1 - yp_train) * P_train)))
            dL_fp_db = (2 * lambda_fp * fp_violation * 
                       (np.mean(d_surv_u_dyu * yu * (1 - yu)) - 
                        eta_p * np.mean(d_surv_p_dyp * yp_train * (1 - yp_train))))
        else:
            dL_fp_dw = 0.0
            dL_fp_db = 0.0
        
        dw = dL_pos_dw + dL_neg_dw + dL_reg_dw + dL_fn_dw + dL_fp_dw
        db = dL_pos_db + dL_neg_db + dL_reg_db + dL_fn_db + dL_fp_db
        
        # ============================================
        # Gradients w.r.t. eta_p (via eta_logit)
        # ============================================
        # From PU loss
        dL_pos_deta = -np.mean(np.log(np.clip(yp_train, 1e-12, 1.0)))
        if L_neg_raw > 0:
            dL_neg_deta = -np.mean(-np.log(np.clip(1 - yp_train, 1e-12, 1.0)))
        else:
            dL_neg_deta = 0.0
        
        # From FP penalty: d(fp_soft)/d(eta_p) = -surv_p
        if fp_violation > 0:
            dL_fp_deta = 2 * lambda_fp * fp_violation * (-surv_p)
        else:
            dL_fp_deta = 0.0
        
        dL_total_deta = dL_pos_deta + dL_neg_deta + dL_fp_deta
        
        # Chain rule: eta_p = sigmoid(eta_logit)
        dL_total_deta_logit = dL_total_deta * eta_p * (1 - eta_p)
        
        # ============================================
        # Gradients w.r.t. tau (via tau_logit)
        # ============================================
        # From FN: d[sigma((tau - y)/temp)]/dtau = sigma'((tau-y)/temp)/temp
        dL_fn_dtau = lambda_fn * np.mean((fn_vals * (1 - fn_vals)) / temp)
        
        # From FP penalty: d[sigma((y - tau)/temp)]/dtau = -sigma'((y-tau)/temp)/temp
        if fp_violation > 0:
            d_surv_u_dtau = -np.mean((surv_u_vals * (1 - surv_u_vals)) / temp)
            d_surv_p_dtau = -np.mean((surv_p_vals * (1 - surv_p_vals)) / temp)
            d_fp_dtau = d_surv_u_dtau - eta_p * d_surv_p_dtau
            dL_fp_dtau = 2 * lambda_fp * fp_violation * d_fp_dtau
        else:
            dL_fp_dtau = 0.0
        
        dL_total_dtau = dL_fn_dtau + dL_fp_dtau
        
        # Chain rule: tau = sigmoid(tau_logit)
        dL_total_dtau_logit = dL_total_dtau * tau * (1 - tau)
        
        # ============================================
        # Updates
        # ============================================
        w -= lr * dw
        b -= lr * db
        eta_logit -= lr * dL_total_deta_logit
        tau_logit -= lr * dL_total_dtau_logit
        
        # Monitoring
        if (epoch + 1) % 500 == 0:
            hard_fn = np.mean(yp_val < tau)
            print(f"Epoch {epoch+1}: eta_p={eta_p:.3f}, tau={tau:.3f}, "
                  f"L_pu={L_pu:.4f}, FN_soft={fn_soft:.4f}, hard_FN={hard_fn:.4f}, "
                  f"FP_soft={fp_soft:.4f}, FP_viol={fp_violation:.4f}")
    
    return w, b, sigmoid(eta_logit), sigmoid(tau_logit)

# ------------------------------
# 3) Train
# ------------------------------
budget = 0.02

print("Training elegant PU with joint optimization...\n")
w_star, b_star, eta_star, tau_star = train_elegant_pu(
    P_train, P_val, U,
    budget=budget,
    lr=1e-2,
    epochs=30000,
    l2=1e-4,
    temp=0.05,
    lambda_fn=10.0,
    lambda_fp=100.0,
    seed=1
)

def compute_fn(w, b, P_eval, tau):
    y = sigmoid(w * P_eval + b)
    return np.mean(y < tau)

fn_star = compute_fn(w_star, b_star, P_val, tau_star)

# ------------------------------
# 4) Visualization
# ------------------------------
xs = np.linspace(-2.5, 5.0, 400)
y_star = sigmoid(w_star * xs + b_star)

fig, ax = plt.subplots(figsize=(8,5))

ax.hist(neg, bins=40, density=True, alpha=0.25, label="Negatives (hidden)", range=(-2.5,5.0))
ax.hist(P_train, bins=40, density=True, alpha=0.25, label="Positives (train)", range=(-2.5,5.0))
ax.hist(U, bins=40, density=True, alpha=0.15, label="Unlabeled mixture", range=(-2.5,5.0))

ax.plot(xs, y_star, label=f"Elegant PU (ηᵖ={eta_star:.2f}), FN={fn_star:.3f}", linewidth=2)
ax.axhline(tau_star, linestyle="--", linewidth=1.5, color='red', 
           label=f"τ* (learned): {tau_star:.2f}")

ax.set_xlabel("1-D score x")
ax.set_ylabel("Predicted P(y=1|x)")
ax.set_title("Elegant PU: Joint optimization of (w, b, ηₚ, τ)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

print("\n=== Results ===")
print(f"True prior π_true = {pi_true:.2f}")
print(f"Elegant PU: eta_p*={eta_star:.2f}, tau*={tau_star:.2f}, FN={fn_star:.4f}")