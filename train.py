import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Dummy data - Real data baad mein daalenge
data = {
    'size': [1000, 1500, 2000, 1200, 1800],
    'bedrooms': [2, 3, 4, 2, 3],
    'bathrooms': [1, 2, 2.5, 1, 2],
    'price': [200000, 300000, 400000, 220000, 350000]
}

df = pd.DataFrame(data)
X = df[['size', 'bedrooms', 'bathrooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
