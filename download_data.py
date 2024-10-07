from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
iris = load_iris()

# Create a pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column to the DataFrame
iris_df['target'] = iris.target

# Save the dataset to a CSV file
iris_df.to_csv('iris_data.csv', index=False)

print("Iris dataset downloaded and saved as 'iris_data.csv'.")
