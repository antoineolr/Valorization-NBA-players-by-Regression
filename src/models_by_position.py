import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score, mean_squared_error

def train_models_by_position(df, X_train_scaled, X_test_scaled, y_train, y_test, train_idx, test_idx):
    """
    Entraîne un modèle OLS et Lasso pour chaque position
    
    Args:
        df: DataFrame original avec POSITION
        X_train_scaled: Features d'entraînement standardisées
        X_test_scaled: Features de test standardisées
        y_train: Salaires d'entraînement (log)
        y_test: Salaires de test (log)
        train_idx: Indices d'entraînement
        test_idx: Indices de test
    
    Returns:
        dict: {position: {'model_ols': model, 'model_lasso': model, 'metrics': {...}}}
    """
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    results = {}
    
    # Récupérer les positions pour train et test
    df_reset = df.reset_index(drop=True)
    train_positions = df_reset.iloc[train_idx]['POSITION'].values
    test_positions = df_reset.iloc[test_idx]['POSITION'].values
    
    print("\n" + "="*60)
    print("🏀 ENTRAÎNEMENT DES MODÈLES PAR POSITION")
    print("="*60)
    
    for pos in positions:
        print(f"\n--- {pos} ---")
        
        # Filtrer les données pour cette position
        train_mask = train_positions == pos
        test_mask = test_positions == pos
        
        X_train_pos = X_train_scaled[train_mask]
        y_train_pos = y_train.iloc[train_mask] if isinstance(y_train, pd.Series) else y_train[train_mask]
        X_test_pos = X_test_scaled[test_mask]
        y_test_pos = y_test.iloc[test_mask] if isinstance(y_test, pd.Series) else y_test[test_mask]
        
        n_train = len(X_train_pos)
        n_test = len(X_test_pos)
        
        print(f"Train: {n_train} joueurs | Test: {n_test} joueurs")
        
        # Vérifier qu'on a assez de données
        if n_train < 10 or n_test < 3:
            print(f"⚠️  Pas assez de données pour {pos}, skip")
            results[pos] = None
            continue
        
        # Entraîner OLS
        model_ols = LinearRegression()
        model_ols.fit(X_train_pos, y_train_pos)
        y_pred_ols = model_ols.predict(X_test_pos)
        
        r2_ols = r2_score(y_test_pos, y_pred_ols)
        rmse_ols = np.sqrt(mean_squared_error(y_test_pos, y_pred_ols))
        
        # Entraîner Lasso
        alphas = np.logspace(-4, 1, 100)
        model_lasso = LassoCV(alphas=alphas, cv=min(5, n_train), random_state=42, max_iter=10000)
        model_lasso.fit(X_train_pos, y_train_pos)
        y_pred_lasso = model_lasso.predict(X_test_pos)
        
        r2_lasso = r2_score(y_test_pos, y_pred_lasso)
        rmse_lasso = np.sqrt(mean_squared_error(y_test_pos, y_pred_lasso))
        
        print(f"OLS   - R²: {r2_ols:.4f}, RMSE: {rmse_ols:.4f}")
        print(f"Lasso - R²: {r2_lasso:.4f}, RMSE: {rmse_lasso:.4f}, α: {model_lasso.alpha_:.6f}")
        
        results[pos] = {
            'model_ols': model_ols,
            'model_lasso': model_lasso,
            'metrics': {
                'r2_ols': r2_ols,
                'rmse_ols': rmse_ols,
                'r2_lasso': r2_lasso,
                'rmse_lasso': rmse_lasso,
                'alpha': model_lasso.alpha_,
                'n_train': n_train,
                'n_test': n_test
            },
            'predictions': {
                'y_test': y_test_pos,
                'y_pred_ols': y_pred_ols,
                'y_pred_lasso': y_pred_lasso
            },
            'indices': {
                'test_mask': test_mask
            }
        }
    
    return results


def compare_feature_importance_by_position(results, feature_names):
    """
    Compare l'importance des features entre les différentes positions
    
    Args:
        results: Résultats de train_models_by_position()
        feature_names: Liste des noms de features
    
    Returns:
        pd.DataFrame: DataFrame avec importance par position
    """
    importance_df = pd.DataFrame(index=feature_names)
    
    for pos, data in results.items():
        if data is not None:
            importance_df[pos] = data['model_lasso'].coef_
    
    # Ajouter une colonne "Global" avec la moyenne
    importance_df['MEAN'] = importance_df.mean(axis=1)
    
    return importance_df


def print_feature_importance_comparison(importance_df, top_n=10):
    """
    Affiche l'importance des features par position
    """
    print("\n" + "="*80)
    print("📊 IMPORTANCE DES FEATURES PAR POSITION (Lasso)")
    print("="*80)
    
    # Trier par importance moyenne
    importance_sorted = importance_df.sort_values('MEAN', ascending=False, key=abs)
    
    print(f"\n{'Feature':<20} {'PG':>8} {'SG':>8} {'SF':>8} {'PF':>8} {'C':>8} {'MEAN':>8}")
    print("-" * 80)
    
    for idx, row in importance_sorted.head(top_n).iterrows():
        values = [f"{row.get(pos, 0):>8.4f}" if pos in row.index else "    -   " 
                  for pos in ['PG', 'SG', 'SF', 'PF', 'C', 'MEAN']]
        print(f"{idx:<20} {' '.join(values)}")
