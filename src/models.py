from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def train_ols(X_train, y_train):
    model=LinearRegression()
    model.fit(X_train,y_train)
    return model

def train_lasso(X_train, y_train):
    model = LassoCV(cv=10, random_state=42, alphas=np.logspace(-4, 1, 100))
    model.fit(X_train,y_train)
    print("meilleur alpha",model.alpha_)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred=model.predict(X_test)
    R2=r2_score(y_test,y_pred)
    MSE=mean_squared_error(y_test,y_pred)
    RMSE=np.sqrt(MSE)
    
    print(f"\n=== {model_name} ===")
    print(f"R² Score: {R2:.4f}")
    print(f"RMSE: {RMSE:.4f}")
    
    return R2, RMSE, y_pred


def get_feature_importance(model, feature_names):
    coeff=model.coef_
    importance=dict(zip(feature_names,coeff))
    sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\n=== Importance des Features ===")
    for feature, coef in sorted_importance:
        print(f"{feature:20s}: {coef:8.4f}")
    
    return sorted_importance