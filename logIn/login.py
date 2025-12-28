import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def predict_login_risk(login_attempt):
    X_train = [
    [2, 8, 5, 1],
    [14, 2, 0, 0],
    [23, 5, 2, 1],
    [9, 1, 0, 0],
    [3, 9, 10, 1],
    [16, 3, 1, 0]
    ]

    y_train = [
        95.0,
        10.5,
        75.8,
        5.0,
        99.5,
        40.0
    ]

    risk = DecisionTreeRegressor(random_state=0)
    risk.fit(X_train, y_train)
    
    score = risk.predict([login_attempt])
    
    return score

print("Enter Login Attempt Details To Calculate Risk Score")
try:
    hour = int(input("Enter the hour of the day (0-23): "))
    country = int(input("Enter the country risk level (1-10): "))
    attempts = int(input("Enter failed attempts in the last hour: "))
    new_device = int(input("Is it a new device? (1 for yes, 0 for no): "))

    a = [hour, country, attempts, new_device]

    risk_score = predict_login_risk(a)

    print(f"\nAttempt Details: {a}")
    print(f"Predicted Risk Score: {risk_score:.1f}/100.0")

except ValueError:
    print("\nInvalid input. Please enter whole numbers only.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")