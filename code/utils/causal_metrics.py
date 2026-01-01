import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def compute_arc_metrics(df, treatment_col, outcome_col, confounders, model_type="linear"):
    """
    Computes Causal Metrics for BINARY Treatment (T=0 vs T=1).
    
    Args:
        model_type: "linear" (Ridge) or "tree" (GradientBoosting). 
    """
    
    # ---------------------------------------------------------
    # 1. Causal Model (Adjusted for Z) -> ATE (Merit)
    # ---------------------------------------------------------
    features = confounders + [treatment_col]
    X = df[features]
    y = df[outcome_col]
    
    if model_type == "tree":
        model_causal = HistGradientBoostingRegressor(max_iter=50, max_depth=5, random_state=42)
    else:
        model_causal = Ridge(alpha=1.0) # Linear model forces a coefficient for T

    ate_causal = 0.0
    
    try:
        model_causal.fit(X, y)
        
        # Counterfactual: Force whole population to T=1
        X_1 = X.copy()
        X_1[treatment_col] = 1
        pred_1 = model_causal.predict(X_1)
        
        # Counterfactual: Force whole population to T=0
        X_0 = X.copy()
        X_0[treatment_col] = 0
        pred_0 = model_causal.predict(X_0)
        
        # ATE = Mean difference
        ate_causal = np.mean(pred_1 - pred_0)
        
    except Exception as e:
        # print(f"Error in ATE calculation: {e}")
        pass

    # ---------------------------------------------------------
    # 2. Naive Model (Ignored Z) -> Naive Association
    # ---------------------------------------------------------
    X_naive = df[[treatment_col]]
    
    if model_type == "tree":
        model_naive = HistGradientBoostingRegressor(max_iter=50, max_depth=5, random_state=42)
    else:
        model_naive = Ridge(alpha=1.0)

    naive_assoc = 0.0
    
    try:
        model_naive.fit(X_naive, y)
        
        X_naive_1 = X_naive.copy()
        X_naive_1[treatment_col] = 1
        pred_naive_1 = model_naive.predict(X_naive_1)
        
        X_naive_0 = X_naive.copy()
        X_naive_0[treatment_col] = 0
        pred_naive_0 = model_naive.predict(X_naive_0)
        
        naive_assoc = np.mean(pred_naive_1 - pred_naive_0)
        
    except Exception as e:
        # print(f"Error in Naive calculation: {e}")
        pass

    # ---------------------------------------------------------
    # 3. Compute DIE (Confounding Bias)
    # ---------------------------------------------------------
    die_confounding = abs(naive_assoc - ate_causal)
    
    return {
        "ATE": ate_causal,
        "Naive_Assoc": naive_assoc,
        "DIE_Confounding": die_confounding
    }

def compute_direct_effect(df, treatment_col, outcome_col, confounders):
    """
    Computes the Direct Effect (ATE) of the Protected Attribute (Z).
    """
    features = confounders + [treatment_col]
    X = df[features]
    y = df[outcome_col]
    
    # Use Ridge for consistency if we switched above, or stick to Tree for non-linearity
    model = Ridge(alpha=1.0) 
    
    try:
        model.fit(X, y)
        
        # We need to know WHICH confounder to flip. 
        # Assuming Z_race is the first one or we flip all Zs?
        # Let's assume we are measuring the effect of the FIRST confounder.
        target_z = confounders[0]
        
        X_0 = X.copy()
        X_0[target_z] = 0
        pred_0 = model.predict(X_0)
        
        X_1 = X.copy()
        X_1[target_z] = 1
        pred_1 = model.predict(X_1)
        
        return np.mean(pred_1 - pred_0)
    except Exception as e:
        return 0.0