import sys
print(sys.version)
sys.path.append('../cmake-build-debug')
import linear_regression

def main():
    # Create an instance of the LinearRegression class with learning rate and iterations
    model = linear_regression.LinearRegression(learning_rate=0.01, iterations=100)

    # Data for fitting
    X = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]

    # Fit the model with data
    model.fit(X, y)

    # Predict some values
    predictions = model.predict([6.0, 7.0, 8.0])

    # Get the slope and intercept
    slope = model.getSlope()
    intercept = model.getIntercept()

    # Print results
    print("Predictions:", predictions)
    print("Slope:", slope)
    print("Intercept:", intercept)

if __name__ == '__main__':
    main()
