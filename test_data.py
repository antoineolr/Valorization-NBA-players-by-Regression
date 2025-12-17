"""
Script de test pour vérifier que data_handler fonctionne
"""
from src.data_handler import fetch_nba_data

# Test de récupération des données (10 minutes par match minimum)
df = fetch_nba_data(season='2023-24', min_minutes_per_game=10)

print("\n" + "="*60)
print("📋 APERÇU DES DONNÉES")
print("="*60)
print(f"\nShape: {df.shape}")
print(f"\nColonnes: {df.columns.tolist()}")

if len(df) > 0:
    print(f"\n10 premiers joueurs:")
    print(df[['PLAYER_NAME', 'MIN', 'GP', 'SALARY']].head(10))
    print(f"\nStatistiques des salaires:")
    print(df['SALARY'].describe())
    print(f"\nStatistiques des minutes par match:")
    print(df['MIN'].describe())
else:
    print("\n⚠️  Aucune donnée disponible")
