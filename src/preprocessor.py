import numpy as np              # Pour log(), calculs mathématiques
import pandas as pd             # Pour manipuler le DataFrame
from sklearn.model_selection import train_test_split  # Pour split train/test
from sklearn.preprocessing import StandardScaler      # Pour standardiser X

def preprocess_data(df):
    y=df['SALARY']
    X=df[['AGE',
        'GP', 'MIN','PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'PLUS_MINUS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING']]
    y_log=np.log(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    scaler = StandardScaler()       
    X_train_scaled = scaler.fit_transform(X_train)  # Applique sur train
    X_test_scaled = scaler.transform(X_test)    # Applique sur test
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def print_preprocessing_info(df, X_train, X_test, y_train, y_test):
    print(f"Dataset original: {len(df)} joueurs")
    print(f"Train set: {len(X_train)} joueurs")
    print(f"Test set: {len(X_test)} joueurs")
    print(f"Features utilisées: {X_train.shape[1]}")

    
