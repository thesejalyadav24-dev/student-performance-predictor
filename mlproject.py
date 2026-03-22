import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [30, 35, 50, 55, 65, 70, 80, 90]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split input & output
X = df[["Hours"]]
y = df["Marks"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Take input4
hours = float(input("Enter study hours: "))

if hours < 0 or hours > 10:
    print("Enter study hours between 0 and 10")
else:
    # CREATE input_data here
    input_data = pd.DataFrame([[hours]], columns=["Hours"])
    
    # NOW predict
    pred = model.predict(input_data)
    
    pred_value = min(pred[0], 100)
    print(f"Predicted Marks: {round(pred_value, 2)}")
    # Plot graph
    plt.scatter(df["Hours"], df["Marks"])
    plt.plot(df["Hours"], model.predict(X))
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.title("Study vs Marks")
    plt.show()