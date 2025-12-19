import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_waste_model(file):
    # Load data
    df = pd.read_csv(file)
    
    # Define features and target
    y = df['waste_kg']
    X = df[['population', 'collection_capacity_kg', 'temp_c', 'rain_mm', 
            'overflow', 'is_weekend', 'is_holiday', 'recycling_campaign', 
            'day_name', 'area']]
    
    # One-Hot Encoding
    X = pd.get_dummies(X, columns=['day_name', 'area'], drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    
    return df, X_test, y_test, y_pred, metrics