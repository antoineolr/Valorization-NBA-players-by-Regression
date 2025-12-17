from src.data_handler import fetch_nba_data
from src.preprocessor import preprocess_data
from src.models import train_ols, train_lasso, evaluate_model, get_feature_importance
from src.analyzer import (calculate_residuals, identify_undervalued_players, 
                          identify_overvalued_players, plot_predictions, 
                          plot_residuals, plot_feature_importance)

MIN_MINUTES_PLAYED = 500  # Minutes totales sur la saison
MIN_MINUTES_PER_GAME = 10  # Minutes par match minimum

df= fetch_nba_data(season='2023-24', min_minutes_per_game=MIN_MINUTES_PER_GAME, csv_path='data/nba_salaries_2023_24.csv')
print("Apercu",df.head())
print("featrures", df.columns.tolist())
print("nombre de joueurs: ", df.shape)

players_name_all= df['PLAYER_NAME'].values
X_train, X_test, y_train, y_test, scaler= preprocess_data(df)
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42
)
player_names_test = players_name_all[test_idx]

model_ols= train_ols(X_train,y_train)
model_lasso=train_lasso(X_train,y_train)

R2_ols, RMSE_ols, y_pred_ols= evaluate_model(model_ols, X_test, y_test, model_name="OLS")
R2_lasso, RMSE_lasso, y_pred_lasso= evaluate_model(model_lasso, X_test, y_test, model_name="LASSO")

feature_names = ['AGE', 'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 
                 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING']
importance=get_feature_importance(model_lasso, feature_names)

salary_pred, salary_real, residuals = calculate_residuals(y_test, y_pred_lasso)

undervalued = identify_undervalued_players(player_names_test, salary_real, salary_pred, residuals, top_n=10)
overvalued = identify_overvalued_players(player_names_test, salary_real, salary_pred, residuals, top_n=10)

plot_predictions(salary_pred, salary_real)
plot_residuals(residuals)
plot_feature_importance(importance, top_n=10)