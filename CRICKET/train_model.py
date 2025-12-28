import yaml
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier # New import for the classifier
from sklearn.metrics import mean_absolute_error, accuracy_score # New import
import pickle
import os

def process_chase_innings(innings_data, target_score): ###each match the game was completely over and one team won
    if 'deliveries' not in innings_data: return [] #empty list
    over_by_over_features = []
    current_score, wickets_lost, total_overs = 0, 0, 20
    deliveries_by_over = {} #HashMap
    for delivery_item in innings_data['deliveries']:
        delivery_key, delivery_data = list(delivery_item.items())[0] #key is like 0.1, data will be batsman, bowler, etc.
        over_num = int(delivery_key) #ex: 0.1
        if over_num not in deliveries_by_over: deliveries_by_over[over_num] = []
        deliveries_by_over[over_num].append(delivery_data) #info is just added hashTable
    for over in sorted(deliveries_by_over.keys()):
        wickets_in_hand = 10 - wickets_lost
        runs_needed = target_score - current_score
        balls_remaining = (total_overs * 6) - (over * 6)
        current_run_rate = (current_score / over) if over > 0 else 0
        overs_remaining = total_overs - over
        required_run_rate = (runs_needed / overs_remaining) if overs_remaining > 0 else float('inf')
        feature_row = {'over': over, 'runs_needed': runs_needed, 'balls_remaining': balls_remaining,
                       'wickets_in_hand': wickets_in_hand, 'current_run_rate': current_run_rate,
                       'required_run_rate': required_run_rate}
        runs_in_this_over = sum(d['runs']['total'] for d in deliveries_by_over[over])
        feature_row['runs_in_over_actual'] = runs_in_this_over
        over_by_over_features.append(feature_row)
        current_score += runs_in_this_over
        for d in deliveries_by_over[over]:
            if 'wicket' in d: wickets_lost += 1
    return over_by_over_features

def create_dataset_from_zip(zip_path):
    """Reads all YAML files and now also extracts the final match winner."""
    all_match_features = []
    print(f"Reading from zip file: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        yaml_files = [f for f in z.namelist() if f.endswith('.yaml')]
        print(f"Found {len(yaml_files)} match files. Processing all of them...")
        for i, filename in enumerate(yaml_files):
            if (i + 1) % 500 == 0: print(f"  Processed {i+1}/{len(yaml_files)} files...")
            try:
                with z.open(filename) as f:
                    match_data = yaml.safe_load(f)
                if 'innings' in match_data and len(match_data['innings']) >= 2:
                    info = match_data.get('info', {})
                    outcome = info.get('outcome', {})
                    if 'winner' in outcome:
                        first_innings = list(match_data['innings'][0].values())[0]
                        second_innings = list(match_data['innings'][1].values())[0]
                        target = None
                        if 'target' in second_innings and 'runs' in second_innings.get('target', {}):
                            target = second_innings['target']['runs']
                        elif 'deliveries' in first_innings:
                            target = sum(list(d.values())[0]['runs']['total'] for d in first_innings['deliveries']) + 1
                        
                        if target:
                            # --- NEW: Determine if the chasing team won ---
                            chasing_team = second_innings['team']
                            winner = outcome['winner']
                            chase_win = 1 if chasing_team == winner else 0
                            
                            match_features = process_chase_innings(second_innings, target)
                            for row in match_features:
                                row['chase_win'] = chase_win # Add the win/loss result to each row
                            all_match_features.extend(match_features)
            except Exception:
                continue
    print("Finished processing all files.")
    return pd.DataFrame(all_match_features)

if __name__ == "__main__":
    ZIP_FILE_PATH = 't20s.zip'
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"Error: Zip file not found at '{ZIP_FILE_PATH}'")
    else:
        dataset = create_dataset_from_zip(ZIP_FILE_PATH)
        if dataset.empty:
            print("\nError: No valid training data could be extracted from the provided files.")
        else:
            print(f"\nTotal training examples (overs): {len(dataset)}")
            dataset.dropna(inplace=True)
            
            features = ['over', 'runs_needed', 'balls_remaining', 'wickets_in_hand',
                        'current_run_rate', 'required_run_rate']
            
            # --- We now have TWO target variables ---
            target_regressor = 'runs_in_over_actual'  # For predicting runs
            target_classifier = 'chase_win'          # For predicting win/loss
            
            X = dataset[features]
            y_reg = dataset[target_regressor]
            y_clf = dataset[target_classifier]

            # Split data for both training processes
            X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
                X, y_reg, y_clf, test_size=0.2, random_state=42)

            # --- 1. Train and save the Run Predictor (Regressor) ---
            print("\nTraining the Run Predictor model...")
            regressor = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
            regressor.fit(X_train, y_reg_train)
            MODEL_REG_FILENAME = 'run_predictor_model.pkl'
            with open(MODEL_REG_FILENAME, 'wb') as file: #writing binary
                pickle.dump(regressor, file)
            print(f"Run Predictor model saved as '{MODEL_REG_FILENAME}'")
            
            # --- 2. Train and save the Win Predictor (Classifier) ---
            print("\nTraining the Win Probability model...")
            classifier = RandomForestClassifier(n_estimators=200, random_state=42)
            classifier.fit(X_train, y_clf_train)
            MODEL_CLF_FILENAME = 'win_predictor_model.pkl'
            with open(MODEL_CLF_FILENAME, 'wb') as file:
                pickle.dump(classifier, file)
            print(f"Win Predictor model saved as '{MODEL_CLF_FILENAME}'")

            # --- Evaluate both models ---
            print("\n--- Model Evaluation ---")
            reg_predictions = regressor.predict(X_test)
            mae = mean_absolute_error(y_reg_test, reg_predictions)
            print(f"Run Predictor (MAE): ~{mae:.2f} runs per over.")

            clf_predictions = classifier.predict(X_test)
            accuracy = accuracy_score(y_clf_test, clf_predictions)
            print(f"Win Predictor (Accuracy): {accuracy:.2%}")