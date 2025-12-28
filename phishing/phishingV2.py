import graphviz
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from rapidfuzz import fuzz
from flask import Flask, request, jsonify

## So u give model the x and the y; the x will be features from the 

"""def diff(s1, s2): #can replace with rapid fuzz but wont get what it is similar too
    if len(s1) < len(s2):
        return diff(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1): #walmart, it does (0,w) (1,a)...
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]"""

def parts_phi(url, brands):
    is_long = int(len(url) > 50)
    has_double_com = int(url.count(".com") >= 2)
    other_tlds = [".net", ".biz", ".irl", ".org", ".io"]
    has_both = int('.com' in url and any(tld in url for tld in other_tlds))
    has_at = int('@' in url)
    has_many_dots = int(url.count('.') > 3)  
    has_hyphen = int('-' in url)
    login_keywords = ["login", "signin", "account", "secure", "verify"]
    has_login_keyword = int(any(keyword in url for keyword in login_keywords))
    
    sus = 0
    freeHostingFinal = 0
    typo = 0
    mightBe = ""
    
    try:
        hostname = url.split('//')[-1].split('/')[0] #after // and before /
        
        subdomain = hostname.split('.')[0]
        if len(subdomain) > 4:
            digits = sum(c.isdigit() for c in subdomain)
            letters = sum(c.isalpha() for c in subdomain)
            if letters == 0 or (letters > 0 and digits / len(subdomain) > 0.5):
                sus = 1
        
        freeHostsSites = ['wcomhost', '000webhost', 'awardspace', 'byethost', 'infinityfree', 
                          'freehostia', 'freehosting', 'godaddysites', 'weebly', 'wixsite', 
                          'yolasite', 'render', 'netlify', 'pages.dev', 'workers.dev', 
                          'surge.sh', 'herokuapp', 'glitch.me', 'duckdns.org', 'no-ip', 
                          'github.io', 'gitlab.io', 'firebaseapp.com', 'web.app', 
                          'pastehtml.com', '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.io']

        freeHostingFinal = int(any(keyword in hostname for keyword in freeHostsSites))

        domain_parts = hostname.split('.')
        if len(domain_parts) > 1:
            main_domain = domain_parts[-2] # e.g., 'walmart' from 'login.walmart.com'
            if "-" in main_domain:
                main_domain = main_domain.split('-')[0]
            for brand in brands:
                ratio = fuzz.ratio(main_domain.lower(), brand)
                if 80 < ratio < 100:
                    typo = 1
                    mightBe = brand
                    break
    except:
        pass 
        a = is_long
    return [is_long, has_double_com, has_both, has_at, has_many_dots, 
            has_hyphen, has_login_keyword, sus, freeHostingFinal, typo], typo, mightBe
X_train = []
y_train = []
brands = []

try:
    # Load the target brands from an external file
    with open('brands.txt', 'r') as f:
        i = 0
        
        for line in f:
            clean_line = line.strip().lower()
            domain_to_process = ""
            domain_to_process = clean_line.split(',')[1].strip()

            brands.append(domain_to_process.split('.')[0]) #manual substring
                

    print(f"Loaded {len(brands)} brands from brands.txt")
except FileNotFoundError:
    print("not found. typo feature will be disabled.")
except Exception as e:
    print(f"An error occurred while reading brands.txt: {e}")

try:
    # only use 10000
    samp = pd.read_csv('a.zip', skiprows=lambda i: i > 0 and (i + 1) % 80 != 0)
    
    df = samp.sample(n=10000, random_state=42) #randomize!!!
    #df = pd.read_csv('a.zip')
    for index, row in df.iterrows(): #we dont use index but need it so no errors
        url = row['url']
        status = row['status']
        
        features, _, ee = parts_phi(url, brands)
        X_train.append(features)
        
        # use 1 for phishing bcause i used 1 while data used 0
        y_train.append(1 if status == 0 else 0)

except FileNotFoundError:
    print("Error: 'a.zip' not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()


phishing_detector = RandomForestClassifier(n_estimators=300, random_state=0)
phishing_detector.fit(X_train, y_train)

"""dotWhat = user_url.split('.')[1]

features, typo, mightBe = parts_phi(user_url, brands)
prediction = phishing_detector.predict([features])

print(f"\nURL: {user_url}")
print(f"Features: {features}") """

feature_names = [
    'is_long', 'has_double_com', 'has_com_and_other_tld', 'has_at', 
    'has_many_dots', 'has_hyphen', 'has_login_keyword', 
    'has_suspicious_subdomain', 'uses_free_hosting', 'is_typosquatting'
]
"""dot_data = tree.export_graphviz(phishing_detector, out_file=None, 
                                feature_names=feature_names,  
                                class_names=['Legitimate', 'Phishing'],  
                                filled=True, rounded=True,  
                                special_characters=True)
graph = graphviz.Source(dot_data)
try:
    graph.render("phishing_tree", cleanup=True)
except graphviz.backend.execute.ExecutableNotFound:
    print("Error")"""


app = Flask(__name__)

#send requests to '/analyze'
@app.route('/analyze', methods=['POST']) #POST will allow it to accept data coming from user
def analyze_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Please provide a 'url' in the JSON body."}), 400
    user_url = data['url']
    features, typo, mightBe = parts_phi(user_url, brands)
    prediction = phishing_detector.predict([features])
    
    dotWhat = user_url.split('.')[1]
    result_text = f"Might be phishing due to similarities with a famous URL's company: {mightBe}" if typo else ("Phishing" if prediction[0] == 1 else "Legitimate")
    
    response_data = {
        "url": user_url,
        "prediction": result_text,
        "features": features
    }
    
    return jsonify(response_data)


#makes the script runnable
if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Invoke-WebRequest -Uri http://127.0.0.1:5000/analyze -Method POST -ContentType "application/json" -Body '{"url": ""}'