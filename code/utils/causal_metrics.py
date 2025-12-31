import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Suppress warnings for cleaner logs
import warnings
warnings.filterwarnings("ignore")

def compute_arc_metrics(df, treatment_col, outcome_col, confounders, increase_pct=0.1):
    """
    Computes Causal Metrics based on the ARC Framework.
    
    Metrics:
    1. ATE_Merit (True Utility): E[Y|do(T+)] - E[Y|do(T)]. 
       The pure causal reward for merit, adjusted for confounders.
    
    2. Naive_Assoc (Observed Utility): E[Y|T+] - E[Y|T].
       The simple correlation, contaminated by confounders.
       
    3. DIE_Confounding (Deconfounding Impact Estimation): | Naive - ATE |.
       Measures the magnitude of the confounder's spurious influence.
       
    Returns: Dictionary with 'ATE_Merit', 'Naive_Assoc', and 'DIE_Confounding'.
    """
    
    # 1. Causal Model (Adjusted for Z) -> Gives ATE (Merit)
    # We control for confounders (Z) to isolate the effect of T.
    features = confounders + [treatment_col]
    X = df[features]
    y = df[outcome_col]
    

    # Use fast HistGradientBoosting
    model_causal = HistGradientBoostingRegressor(max_iter=50, max_depth=5, random_state=42)
    ate_merit = 0.0
    
    try:
        model_causal.fit(X, y)
        
        # Predict Natural State
        risk_now = model_causal.predict(X)
        
        # Predict Counterfactual (Intervention on T)
        X_cf = X.copy()
        # T_new = T * 1.1 (Increase Merit by 10%), clipped to valid range [0, 1]
        X_cf[treatment_col] = np.clip(X_cf[treatment_col] * (1 + increase_pct), 0, 1)
        
        risk_cf = model_causal.predict(X_cf)
        
        ate_merit = np.mean(risk_cf - risk_now)
    except Exception as e:
        print(f"Error in ATE Merit calculation: {e}")
        pass

    # 2. Naive Model (Ignored Z) -> Gives Naive Association
    # We IGNORE confounders (Z) and look only at T -> Y.
    X_naive = df[[treatment_col]]
    
    model_naive = HistGradientBoostingRegressor(max_iter=50, max_depth=5, random_state=42)
    naive_assoc = 0.0
    
    try:
        model_naive.fit(X_naive, y)
        
        risk_now_naive = model_naive.predict(X_naive)
        
        X_cf_naive = X_naive.copy()
        X_cf_naive[treatment_col] = np.clip(X_cf_naive[treatment_col] * (1 + increase_pct), 0, 1)
        
        risk_cf_naive = model_naive.predict(X_cf_naive)
        
        naive_assoc = np.mean(risk_cf_naive - risk_now_naive)
    except Exception as e:
        pass

    # 3. Compute DIE (The Gap)
    # Ideally, we want Naive to equal Causal (meaning Z doesn't distort the view of T).
    die_confounding = abs(naive_assoc - ate_merit)
    
    return {
        "ATE_Merit": ate_merit,
        "Naive_Assoc": naive_assoc,
        "DIE_Confounding": die_confounding
    }

def compute_direct_effect(df, treatment_col, outcome_col, confounders):
    """
    Computes the Direct Effect (ATE) of the Protected Attribute (Z).
    E[Y|do(Z=1)] - E[Y|do(Z=0)].
    We want this to be close to 0 for fairness.
    """
    features = confounders + [treatment_col]
    X = df[features]
    y = df[outcome_col]
    
    model = HistGradientBoostingRegressor(max_iter=50, max_depth=5, random_state=42)
    
    try:
        model.fit(X, y)
        
        # Counterfactual 1: Everyone is Group 0
        X_0 = X.copy()
        X_0[treatment_col] = 0
        pred_0 = model.predict(X_0)
        
        # Counterfactual 2: Everyone is Group 1
        X_1 = X.copy()
        X_1[treatment_col] = 1
        pred_1 = model.predict(X_1)
        
        # ATE is the average difference
        return np.mean(pred_1 - pred_0)
    except Exception as e:
        print(f"Error in Direct Effect calculation: {e}")
        return 0.0

# Legacy alias for backward compatibility (if needed by old notebooks/scripts)
# This maps the old function name to ATE_Merit (Utility)
def compute_die(df, treatment_col, outcome_col, confounders, increase_pct=0.1):
    return compute_arc_metrics(df, treatment_col, outcome_col, confounders, increase_pct)['ATE_Merit']