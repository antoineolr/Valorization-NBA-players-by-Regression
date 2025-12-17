"""
Debug script pour voir les données brutes de l'API
"""
from nba_api.stats.endpoints import leaguedashplayerstats

stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season='2023-24',
    measure_type_detailed_defense='Advanced',
    per_mode_detailed='Totals'
)

df = stats.get_data_frames()[0]

print("Colonnes disponibles:")
print(df.columns.tolist())
print("\n5 premières lignes:")
print(df[['PLAYER_NAME', 'GP', 'MIN']].head(10) if 'MIN' in df.columns else df[['PLAYER_NAME', 'GP']].head(10))
