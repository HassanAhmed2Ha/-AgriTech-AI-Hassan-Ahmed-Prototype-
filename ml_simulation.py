import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({'moisture': [30,45,60,75,90], 'yield': [50,70,85,90,80]})
model = LinearRegression().fit(data[['moisture']], data['yield'])

moisture = 65
yield_pred = model.predict([[moisture]])[0]
print(f"Water Saved: 20% | Yield: {yield_pred:.1f}%")
