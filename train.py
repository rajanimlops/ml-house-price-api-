import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

df = pd.read_csv("train.csv")
X = df[['size', 'bedrooms', 'bathrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"R2 Score: {r2:.2f}")
print(f"Average Error: {mae:.0f} rupees")
print(f"Formula: Price = {model.coef_[0]:.0f}*size + {model.coef_[1]:.0f}*bedrooms + {model.coef_[2]:.0f}*bathrooms + {model.intercept_:.0f}")

pickle.dump(model, open("model.pkl", "wb"))
print("model.pkl saved")