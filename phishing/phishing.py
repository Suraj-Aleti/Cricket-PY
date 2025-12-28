from sklearn.tree import DecisionTreeClassifier

user = input("Enter you url: ")

X_train = [
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

y_train = [1, 1, 1, 1, 1, 0, 0, 0, 0]

phishing_detector = DecisionTreeClassifier(random_state=0)
phishing_detector.fit(X_train, y_train)


def extract_features(url):
    is_long = 1 if len(url) > 50 else 0

    has_com = int(url.count(".com") >= 2)

    other = [".net", ".biz", ".irl", ".org", ".io"]
    has_both = int('.com' in url and any(option in url for option in other))

    has_at = 1 if '@' in url else 0
    
    has_many_dots = 1 if url.count('.') > 3 else 0
    
    return [is_long, has_com, has_both, has_at, has_many_dots]



print(f"Testing URL: {user}")
    
features = extract_features(user)
print(f"Extracted Features: {features}")
    
prediction = phishing_detector.predict([features])
    
if prediction[0] == 1:
    print("Result: PHISHING")
else:
    print("Result: Legitimate")