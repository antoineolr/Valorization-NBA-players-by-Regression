import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leaguedashplayerstats, commonallplayers
from nba_api.stats.static import players
import time

MIN_MINUTES_PLAYED = 500  # Minutes totales sur la saison
MIN_MINUTES_PER_GAME = 10  # Minutes par match minimum

def fetch_nba_stats(season='2023-24'):
    """
    Récupère les statistiques avancées des joueurs NBA via l'API officielle
    
    Args:
        season (str): Saison au format '2023-24'
    
    Returns:
        pd.DataFrame: DataFrame avec les stats des joueurs
    """
    print(f"📊 Récupération des statistiques NBA pour la saison {season}...")
    
    # Récupération des stats de base
    stats_base = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame'
    )
    df_base = stats_base.get_data_frames()[0]
    
    # Récupération des stats avancées
    stats_advanced = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame'
    )
    df_advanced = stats_advanced.get_data_frames()[0]
    
    # Récupération des positions des joueurs
    print(f"🏀 Récupération des positions des joueurs...")
    all_players = commonallplayers.CommonAllPlayers(
        is_only_current_season=1,
        league_id='00',
        season=season
    )
    df_positions = all_players.get_data_frames()[0]
    df_positions = df_positions[['PERSON_ID', 'DISPLAY_FIRST_LAST', 'ROSTERSTATUS', 'FROM_YEAR', 'TO_YEAR']]
    
    # Note: L'API commonallplayers ne retourne pas toujours POSITION
    # On va utiliser une approche alternative avec les stats de base qui contiennent parfois POSITION
    
    # Colonnes à garder de chaque DataFrame
    base_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'AGE',
                 'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS']
    
    advanced_cols = ['PLAYER_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING']
    
    # Garder seulement les colonnes disponibles
    base_cols = [col for col in base_cols if col in df_base.columns]
    advanced_cols = [col for col in advanced_cols if col in df_advanced.columns]
    
    df_base = df_base[base_cols]
    df_advanced = df_advanced[advanced_cols]
    
    # Fusionner les deux DataFrames
    df = df_base.merge(df_advanced, on='PLAYER_ID', how='left')
    
    # Récupérer les positions depuis nba_api.stats.static
    print(f"🔍 Ajout des positions...")
    all_players_static = players.get_players()
    position_dict = {p['id']: p.get('position', 'Unknown') for p in all_players_static if 'position' in p}
    
    # Si la position n'est pas disponible dans static, on utilise une approche heuristique
    # basée sur les statistiques (en dernier recours)
    df['POSITION'] = df['PLAYER_ID'].map(position_dict)
    
    # Pour les joueurs sans position, on va utiliser une heuristique simple
    # basée sur leurs statistiques
    df['POSITION'] = df.apply(lambda row: infer_position(row) if pd.isna(row['POSITION']) or row['POSITION'] == 'Unknown' else row['POSITION'], axis=1)
    
    print(f"✅ {len(df)} joueurs récupérés")
    return df


def infer_position(row):
    """
    Infère la position d'un joueur basé sur ses statistiques
    Heuristique simple basée sur les ratios de stats
    """
    # Si on n'a pas assez de données
    if pd.isna(row.get('PTS')) or pd.isna(row.get('REB')) or pd.isna(row.get('AST')):
        return 'F'  # Forward par défaut
    
    pts = row.get('PTS', 0)
    reb = row.get('REB', 0)
    ast = row.get('AST', 0)
    blk = row.get('BLK', 0)
    
    # Pivot (C): Beaucoup de rebonds et blocks
    if reb > 8 and blk > 1:
        return 'C'
    # Meneur (PG): Beaucoup de passes
    elif ast > 5:
        return 'PG'
    # Ailier fort (PF): Rebonds moyens
    elif reb > 6:
        return 'PF'
    # Arrière (SG): Points moyens/élevés, peu de passes
    elif ast < 3 and pts > 12:
        return 'SG'
    # Ailier (SF): Par défaut
    else:
        return 'SF'


def load_salaries_from_csv(csv_path='data/nba_salaries_2023_24.csv'):
    """
    Charge les salaires depuis un fichier CSV
    
    Args:
        csv_path (str): Chemin vers le fichier CSV des salaires
    
    Returns:
        pd.DataFrame: DataFrame avec PLAYER_NAME et SALARY
    """
    print(f"💰 Chargement des salaires depuis {csv_path}...")
    
    try:
        df_salaries = pd.read_csv(csv_path)
        
        # Vérifier que les colonnes nécessaires existent
        if 'PLAYER_NAME' not in df_salaries.columns or 'SALARY' not in df_salaries.columns:
            print("❌ Le fichier CSV doit contenir les colonnes 'PLAYER_NAME' et 'SALARY'")
            return pd.DataFrame()
        
        # Nettoyer les données
        df_salaries = df_salaries.dropna(subset=['PLAYER_NAME', 'SALARY'])
        df_salaries['SALARY'] = df_salaries['SALARY'].astype(float)
        
        print(f"✅ {len(df_salaries)} salaires chargés")
        return df_salaries
    
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return pd.DataFrame()


def fetch_nba_data(season='2023-24', min_minutes_per_game=MIN_MINUTES_PER_GAME, csv_path='data/nba_salaries_2023_24.csv'):
    """
    Fonction principale : récupère stats + salaires et les fusionne
    
    Args:
        season (str): Saison au format '2023-24'
        min_minutes_per_game (float): Minutes minimales par match pour filtrer les joueurs
        csv_path (str): Chemin vers le fichier CSV des salaires
    
    Returns:
        pd.DataFrame: DataFrame complet avec stats et salaires
    """
    print("=" * 60)
    print("🏀 RÉCUPÉRATION DES DONNÉES NBA")
    print("=" * 60)
    
    # 1. Récupérer les stats
    df_stats = fetch_nba_stats(season)
    
    # 2. Charger les salaires depuis CSV
    df_salaries = load_salaries_from_csv(csv_path)
    
    if df_salaries.empty:
        print("⚠️  Aucun salaire récupéré, impossible de continuer")
        return pd.DataFrame()
    
    # 3. Fusionner les deux DataFrames
    print("\n🔗 Fusion des données...")
    df = df_stats.merge(df_salaries, on='PLAYER_NAME', how='inner')
    
    print(f"✅ {len(df)} joueurs après fusion")
    
    # 4. Filtrer par minutes jouées par match
    if 'MIN' in df.columns:
        df = df[df['MIN'] >= min_minutes_per_game]
        print(f"🔍 Filtrage: {len(df)} joueurs avec au moins {min_minutes_per_game} min/match")
    
    # 5. Nettoyer les valeurs manquantes
    df = df.dropna(subset=['SALARY'])
    
    print(f"\n✅ DONNÉES FINALES: {len(df)} joueurs avec stats et salaires")
    print("=" * 60)
    
    return df
