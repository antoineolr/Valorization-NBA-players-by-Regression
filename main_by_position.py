from src.data_handler import fetch_nba_data
from src.preprocessor import preprocess_data
from src.models_by_position import (train_models_by_position, 
                                     compare_feature_importance_by_position,
                                     print_feature_importance_comparison)
from src.analyzer import (analyze_by_position, 
                          plot_feature_importance_by_position,
                          plot_over_under_valued_by_position)
from sklearn.model_selection import train_test_split

MIN_MINUTES_PER_GAME = 10  # Minutes par match minimum

print("\n" + "="*80)
print("🏀 ANALYSE NBA PAR POSITION")
print("="*80)

# Étape 1: Charger les données
df = fetch_nba_data(season='2023-24', 
                    min_minutes_per_game=MIN_MINUTES_PER_GAME, 
                    csv_path='data/nba_salaries_2023_24.csv')

print(f"\n📊 Distribution des positions:")
print(df['POSITION'].value_counts().sort_index())

# Étape 2: Préprocessing
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Obtenir les indices train/test pour mapper les positions
indices = range(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# Étape 3: Entraîner les modèles par position
results_by_pos = train_models_by_position(
    df, X_train, X_test, y_train, y_test, train_idx, test_idx
)

# Étape 4: Comparer l'importance des features entre postes
feature_names = ['AGE', 'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 
                 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING']

importance_df = compare_feature_importance_by_position(results_by_pos, feature_names)
print_feature_importance_comparison(importance_df, top_n=15)

# Étape 5: Analyser les joueurs sous/sur-évalués par position
analyze_by_position(results_by_pos, df, test_idx, top_n=5)

# Étape 6: Visualisations
print("\n📈 Génération des graphiques...")
plot_feature_importance_by_position(importance_df, top_n=10)
plot_over_under_valued_by_position(results_by_pos, df, test_idx, top_n=5)

print("\n" + "="*80)
print("✅ ANALYSE TERMINÉE")
print("="*80)
