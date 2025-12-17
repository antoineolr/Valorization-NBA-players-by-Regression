import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_residuals(y_true_log,y_pred_log):
    # Convertir en numpy arrays si ce sont des pandas Series
    if hasattr(y_true_log, 'values'):
        y_true_log = y_true_log.values
    if hasattr(y_pred_log, 'values'):
        y_pred_log = y_pred_log.values
    
    salary_real=np.exp(y_true_log)
    salary_pred=np.exp(y_pred_log)
    residuals= salary_real - salary_pred
    return salary_pred, salary_real, residuals

def identify_undervalued_players(player_names, salary_real, salary_pred, residuals, top_n=20):
    """
    Identifie les joueurs sous-évalués (salaire réel < salaire prédit)
    
    Args:
        player_names: Noms des joueurs du test set
        salary_real: Salaires réels (après exp)
        salary_pred: Salaires prédits (après exp)
        residuals: Résidus (real - pred)
        top_n: Nombre de joueurs à retourner
        
    Returns:
        DataFrame avec les joueurs les plus sous-évalués
    """
    # Créer un DataFrame avec toutes les infos
    df = pd.DataFrame({
        'PLAYER_NAME': player_names,
        'SALARY_REAL': salary_real,
        'SALARY_PRED': salary_pred,
        'RESIDUAL': residuals
    })
    
    # Trier par résidus croissants (les plus négatifs = plus sous-évalués)
    df_sorted = df.sort_values('RESIDUAL', ascending=True)
    
    # Afficher les résultats
    print(f"\n{'='*60}")
    print(f"TOP {top_n} JOUEURS SOUS-ÉVALUÉS (Bonnes Affaires)")
    print(f"{'='*60}")
    print(f"{'Joueur':<25} {'Réel':>12} {'Prédit':>12} {'Écart':>12}")
    print(f"{'-'*60}")
    
    for idx, row in df_sorted.head(top_n).iterrows():
        print(f"{row['PLAYER_NAME']:<25} ${row['SALARY_REAL']:>11,.0f} ${row['SALARY_PRED']:>11,.0f} ${row['RESIDUAL']:>11,.0f}")
    
    return df_sorted.head(top_n)


def identify_overvalued_players(player_names, salary_real, salary_pred, residuals, top_n=20):
    """
    Identifie les joueurs sur-évalués (salaire réel > salaire prédit)
    
    Args:
        player_names: Noms des joueurs du test set
        salary_real: Salaires réels (après exp)
        salary_pred: Salaires prédits (après exp)
        residuals: Résidus (real - pred)
        top_n: Nombre de joueurs à retourner
        
    Returns:
        DataFrame avec les joueurs les plus sur-évalués
    """
    # Créer un DataFrame avec toutes les infos
    df = pd.DataFrame({
        'PLAYER_NAME': player_names,
        'SALARY_REAL': salary_real,
        'SALARY_PRED': salary_pred,
        'RESIDUAL': residuals
    })
    
    # Trier par résidus décroissants (les plus positifs = plus sur-évalués)
    df_sorted = df.sort_values('RESIDUAL', ascending=False)
    
    # Afficher les résultats
    print(f"\n{'='*60}")
    print(f"TOP {top_n} JOUEURS SUR-ÉVALUÉS (Sur-payés)")
    print(f"{'='*60}")
    print(f"{'Joueur':<25} {'Réel':>12} {'Prédit':>12} {'Écart':>12}")
    print(f"{'-'*60}")
    
    for idx, row in df_sorted.head(top_n).iterrows():
        print(f"{row['PLAYER_NAME']:<25} ${row['SALARY_REAL']:>11,.0f} ${row['SALARY_PRED']:>11,.0f} ${row['RESIDUAL']:>11,.0f}")
    
    return df_sorted.head(top_n)

def plot_predictions(salary_pred, salary_real):
    min_val = min(salary_pred.min(), salary_real.min())
    max_val = max(salary_pred.max(), salary_real.max())
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot des prédictions
    plt.scatter(salary_pred, salary_real, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Ligne de prédiction parfaite (y = x)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
    
    # Labels et titre
    plt.xlabel('Salaire Prédit ($)', fontsize=12)
    plt.ylabel('Salaire Réel ($)', fontsize=12)
    plt.title('Salaire Réel vs Salaire Prédit', fontsize=14, fontweight='bold')
    
    # Formater l'axe pour afficher en millions
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_residuals(residuals):
    plt.figure(figsize=(10,6))
    plt.hist(residuals, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Résidu = 0')

    plt.xlabel('Résidus ($)', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.title('Distribution des Résidus', fontsize=14, fontweight='bold')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))  

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_feature_importance(importance_dict, top_n=10):
    """
    Graphique en barres des features les plus importantes
    
    Args:
        importance_dict: Liste de tuples (feature_name, coefficient) triée
        top_n: Nombre de features à afficher
    """
    top_features = importance_dict[:top_n]
    
    # Extraire les noms (premier élément de chaque tuple)
    features = [item[0] for item in top_features]
    
    # Extraire les coefficients en valeur absolue (deuxième élément)
    coeffs = [abs(item[1]) for item in top_features]
    
    # Créer le bar chart horizontal
    plt.figure(figsize=(10, 6))
    plt.barh(features, coeffs, color='steelblue', edgecolor='black')
    
    # Inverser l'axe Y (pour avoir le plus important en haut)
    plt.gca().invert_yaxis()
    
    # Labels et titre
    plt.xlabel('Importance (valeur absolue du coefficient)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Importance des Features dans le Modèle Lasso', fontsize=14, fontweight='bold')
    
    # Finitions
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_over_under_valued_by_position(results_by_pos, df, test_idx, top_n=5):
    """
    Graphique des joueurs sur/sous-évalués par position
    
    Args:
        results_by_pos: Résultats de train_models_by_position()
        df: DataFrame original
        test_idx: Indices du test set
        top_n: Nombre de joueurs à afficher par position
    """
    df_reset = df.reset_index(drop=True)
    df_test = df_reset.iloc[test_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    positions_to_plot = ['PG', 'SF', 'PF', 'SG']
    colors = {'undervalued': '#2ecc71', 'overvalued': '#e74c3c'}
    
    plot_idx = 0
    for pos in positions_to_plot:
        if pos not in results_by_pos or results_by_pos[pos] is None:
            continue
            
        data = results_by_pos[pos]
        test_mask = data['indices']['test_mask']
        df_pos = df_test[test_mask].copy()
        
        # Calculer les résidus
        salary_pred, salary_real, residuals = calculate_residuals(
            data['predictions']['y_test'],
            data['predictions']['y_pred_lasso']
        )
        
        df_pos = df_pos.reset_index(drop=True)
        df_pos['SALARY_PRED'] = salary_pred
        df_pos['SALARY_REAL'] = salary_real
        df_pos['RESIDUAL'] = residuals
        
        # Prendre les top sous-évalués et sur-évalués
        df_under = df_pos.sort_values('RESIDUAL', ascending=True).head(top_n)
        df_over = df_pos.sort_values('RESIDUAL', ascending=False).head(top_n)
        
        ax = axes[plot_idx]
        
        # Barres pour sous-évalués (résidus négatifs)
        y_pos_under = np.arange(len(df_under))
        ax.barh(y_pos_under, df_under['RESIDUAL'] / 1e6, 
                color=colors['undervalued'], alpha=0.8, label='Sous-évalué')
        
        # Barres pour sur-évalués (résidus positifs)
        y_pos_over = np.arange(len(df_under), len(df_under) + len(df_over))
        ax.barh(y_pos_over, df_over['RESIDUAL'] / 1e6, 
                color=colors['overvalued'], alpha=0.8, label='Sur-évalué')
        
        # Labels des joueurs
        all_names = list(df_under['PLAYER_NAME']) + list(df_over['PLAYER_NAME'])
        ax.set_yticks(np.arange(len(all_names)))
        ax.set_yticklabels(all_names, fontsize=9)
        
        # Ligne à zéro
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        
        # Labels et titre
        ax.set_xlabel('Écart Salaire (M$)', fontsize=10, fontweight='bold')
        ax.set_title(f'{pos} - Top {top_n} Sous/Sur-évalués', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='best', fontsize=9)
        
        plot_idx += 1
    
    # Supprimer les axes non utilisés
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


def analyze_by_position(results_by_pos, df, test_idx, top_n=5):
    """
    Analyse les joueurs sous/sur-évalués pour chaque position
    
    Args:
        results_by_pos: Résultats de train_models_by_position()
        df: DataFrame original
        test_idx: Indices du test set
        top_n: Nombre de joueurs à afficher par position
    """
    df_reset = df.reset_index(drop=True)
    df_test = df_reset.iloc[test_idx]
    
    print("\n" + "="*80)
    print("🎯 ANALYSE PAR POSITION")
    print("="*80)
    
    for pos, data in results_by_pos.items():
        if data is None:
            continue
        
        # Récupérer les données pour cette position
        test_mask = data['indices']['test_mask']
        df_pos = df_test[test_mask].copy()
        
        # Calculer les résidus
        salary_pred, salary_real, residuals = calculate_residuals(
            data['predictions']['y_test'],
            data['predictions']['y_pred_lasso']
        )
        
        # Reset index pour aligner les arrays
        df_pos = df_pos.reset_index(drop=True)
        df_pos['SALARY_PRED'] = salary_pred
        df_pos['SALARY_REAL'] = salary_real
        df_pos['RESIDUAL'] = residuals
        
        # Sous-évalués
        print(f"\n--- {pos}: TOP {top_n} SOUS-ÉVALUÉS ---")
        df_under = df_pos.sort_values('RESIDUAL', ascending=True).head(top_n)
        print(f"{'Joueur':<20} {'Réel':>12} {'Prédit':>12} {'Écart':>12}")
        print("-" * 60)
        for _, row in df_under.iterrows():
            print(f"{row['PLAYER_NAME']:<20} ${row['SALARY_REAL']:>11,.0f} ${row['SALARY_PRED']:>11,.0f} ${row['RESIDUAL']:>11,.0f}")
        
        # Sur-évalués
        print(f"\n--- {pos}: TOP {top_n} SUR-ÉVALUÉS ---")
        df_over = df_pos.sort_values('RESIDUAL', ascending=False).head(top_n)
        print(f"{'Joueur':<20} {'Réel':>12} {'Prédit':>12} {'Écart':>12}")
        print("-" * 60)
        for _, row in df_over.iterrows():
            print(f"{row['PLAYER_NAME']:<20} ${row['SALARY_REAL']:>11,.0f} ${row['SALARY_PRED']:>11,.0f} ${row['RESIDUAL']:>11,.0f}")


def plot_feature_importance_by_position(importance_df, top_n=10):
    """
    Graphique comparant l'importance des features entre postes
    
    Args:
        importance_df: DataFrame avec importance par position (de compare_feature_importance_by_position)
        top_n: Nombre de features à afficher
    """
    # Sélectionner les top N features selon la moyenne
    top_features = importance_df.reindex(
        importance_df['MEAN'].abs().sort_values(ascending=False).index
    ).head(top_n)
    
    # Retirer la colonne MEAN pour le plot
    plot_data = top_features.drop('MEAN', axis=1)
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(plot_data.index))
    width = 0.15
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (pos, color) in enumerate(zip(positions, colors)):
        if pos in plot_data.columns:
            offset = width * (i - 2)
            ax.bar(x + offset, plot_data[pos], width, label=pos, color=color, edgecolor='black')
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient Lasso', fontsize=12, fontweight='bold')
    ax.set_title('Importance des Features par Position', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.show()
