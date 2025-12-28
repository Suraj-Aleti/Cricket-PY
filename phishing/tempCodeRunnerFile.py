import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def extract_features(url):
    is_long = int(len(url) > 50)
    has_double_com = int(url.count(".com") >= 2)
    other_tlds = [".net", ".biz", ".irl", ".org", ".io"]
    has_com_and_other_tld = int('.com' in url and any(tld in url for tld in other_tlds))
    has_at = int('@' in url)
    has_many_dots = int(url.count('.') > 3)  
    has_hyphen = int('-' in url)
    login_keywords = ["login", "signin", "account", "secure", "verify"]
    has_login_keyword = int(any(keyword in url for keyword in login_keywords))
    
    return [is_long, has_double_com, has_com_and_other_tld, has_at, has_many_dots, has_hyphen, has_login_keyword]

X_train = []
y_train = []


try:
    # only use 10000
    samp = pd.read_csv('a.zip', skiprows=lambda i: i > 0 and (i + 1) % 80 != 0)

    df = samp.sample(n=10000, random_state=42)
    
    for index, row in df.iterrows(): #we dont use index but need it so no errors
        url = row['url']
        status = row['status']
        
        features = extract_features(url)
        X_train.append(features)
        
        # use 1 for phishing bcause i used 1 while data used 0
        y_train.append(1 if status == 0 else 0)

except FileNotFoundError:
    print("Error: 'a.zip' not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()


phishing_detector = DecisionTreeClassifier(random_state=0)
phishing_detector.fit(X_train, y_train)

user_url = input("Enter a URL to check: ")

features = extract_features(user_url)
prediction = phishing_detector.predict([features])

print(f"\nURL: {user_url}")
#print(f"Features: {features}")

if prediction[0] == 1:
    print("Result: PHISHING")
else:
    print("Result: Legitimate")
