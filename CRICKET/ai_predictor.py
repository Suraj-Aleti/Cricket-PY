# ai_predictor.py (v9 - Final with Consistent Hybrid Probability)

import pickle
import pandas as pd
import os

def load_models(reg_filename='run_predictor_model.pkl', clf_filename='win_predictor_model.pkl'):
    """Loads both the regressor and classifier models."""
    if not os.path.exists(reg_filename) or not os.path.exists(clf_filename):
        print(f"Error: Model files not found. Please run the training script that creates both model files.")
        return None, None
    
    with open(reg_filename, 'rb') as f:
        reg_model = pickle.load(f)
    with open(clf_filename, 'rb') as f:
        clf_model = pickle.load(f)
        
    print("Both AI models loaded successfully.")
    return reg_model, clf_model

def get_user_selection(prompt, choices):
    """Helper function to get validated input from a user."""
    print(prompt)
    for i, choice in enumerate(choices, 1): print(f"  {i}. {choice}")
    while True:
        try:
            choice_num = int(input("Enter your choice (number): "))
            if 1 <= choice_num <= len(choices): return choices[choice_num - 1]
            else: print("Invalid choice.")
        except ValueError: print("Invalid input.")

def display_win_prob_chart(batting_prob):
    """Creates a simple text-based chart for win probability."""
    batting_prob = max(0.05, min(0.95, batting_prob))
    bowling_prob = 1 - batting_prob
    
    batting_bar = '█' * int(batting_prob * 40)
    bowling_bar = '░' * (40 - len(batting_bar))
    
    print("--- WIN PROBABILITY (CONFIDENCE-ADJUSTED) ---")
    print(f"Batting: {batting_prob:.1%} | Bowling: {bowling_prob:.1%}")
    print(f"[{batting_bar}{bowling_bar}]")
    print("-" * 41)

def run_ai_prediction_calculator():
    regressor, classifier = load_models()
    if not all([regressor, classifier]): return

    try:
        print("\n--- Enter Live T20 Match Details ---")
        target_score = int(input("Enter the target score to chase: "))
        current_score = int(input("Enter the current score: "))
        overs_completed = int(input("Enter the number of overs completed: "))
        wickets_lost = int(input("Enter the number of wickets lost: "))
        print("\n--- Enter Pitch & Environmental Conditions ---")
        compaction = get_user_selection("1. Pitch Compaction:", ["Hard (Good for Batsmen)", "Soft (Tough for Batsmen)"])
        grass = get_user_selection("2. Grass Cover:", ["Grassy (Helps Seam Bowlers)", "Worn / Bare (Helps Batsmen)"])
        moisture = get_user_selection("3. Pitch Moisture:", ["Dry / Cracked (Aids spinners)", "Damp / Sticky (Unpredictable, tough for Batsmen)", "Normal (No significant moisture effect)"])
        dew_factor = get_user_selection("4. Dew Factor:", ["None", "Light", "Heavy"])

    except ValueError:
        print("\nError: Please enter valid numbers.")
        return

    # --- Step 1: Calculate the AI's Baseline Probability ---
    runs_to_get = target_score - current_score
    total_overs = 20
    balls_remaining = total_overs * 6 - (overs_completed * 6)
    wickets_in_hand = 10 - wickets_lost
    current_run_rate = (current_score / overs_completed) if overs_completed > 0 else 0
    overs_left = total_overs - overs_completed
    required_rate_to_win = (runs_to_get / overs_left) if overs_left > 0 else 999
    
    current_features = pd.DataFrame([[overs_completed, runs_to_get, balls_remaining, wickets_in_hand, current_run_rate, required_rate_to_win]],
                                    columns=['over', 'runs_needed', 'balls_remaining', 'wickets_in_hand', 'current_run_rate', 'required_run_rate'])
    
    win_probability_base = classifier.predict_proba(current_features)[0][1]
    
    # --- Step 2: Run a silent simulation to determine the most likely outcome ---
    over_weights = []
    simulated_score = current_score
    if overs_left > 0:
        for over in range(overs_completed + 1, total_overs + 1):
            runs_needed = target_score - simulated_score
            balls_remaining_loop = (total_overs * 6) - (over * 6)
            current_run_rate_loop = (simulated_score / over) if over > 0 else 0
            overs_rem_loop = total_overs - over
            req_rate_loop = (runs_needed / overs_rem_loop) if overs_rem_loop > 0 else runs_needed

            live_features = pd.DataFrame([[over, runs_needed, balls_remaining_loop, wickets_in_hand, current_run_rate_loop, req_rate_loop]],
                                         columns=['over', 'runs_needed', 'balls_remaining', 'wickets_in_hand', 'current_run_rate', 'required_run_rate'])
            base_prediction = regressor.predict(live_features)[0]
            
            # Adjust the prediction for conditions for the simulation
            if "Hard" in compaction: base_prediction *= 1.05
            if "Soft" in compaction: base_prediction *= 0.95
            if "Grassy" in grass: base_prediction *= 0.95
            if "Worn / Bare" in grass: base_prediction *= 1.05
            if "Dry / Cracked" in moisture: base_prediction *= 0.95
            if "Damp / Sticky" in moisture: base_prediction *= 0.92
            if dew_factor == 'Light': base_prediction *= 1.06
            elif dew_factor == 'Heavy': base_prediction *= 1.14

            over_weights.append(base_prediction)
            simulated_score += base_prediction
            
    simulation_predicts_win = (simulated_score >= target_score)

    # --- Step 3: Create the Hybrid Probability ---
    prob_with_conditions = win_probability_base
    if "Hard" in compaction: prob_with_conditions += 0.04
    if "Soft" in compaction: prob_with_conditions -= 0.05
    if "Grassy" in grass: prob_with_conditions -= 0.06
    if "Worn / Bare" in grass: prob_with_conditions += 0.04
    if "Dry / Cracked" in moisture: prob_with_conditions -= 0.07
    if "Damp / Sticky" in moisture: prob_with_conditions -= 0.08
    if dew_factor == 'Light': prob_with_conditions += 0.05
    elif dew_factor == 'Heavy': prob_with_conditions += 0.12

    # Now, blend with the simulation outcome
    if simulation_predicts_win:
        # If simulation predicts a win, the final probability should be at least 50%
        final_win_probability = max(prob_with_conditions, 0.51) 
    else:
        # If simulation predicts a loss, the final probability should be at most 50%
        final_win_probability = min(prob_with_conditions, 0.49)

    # --- Step 4: Final Display ---
    print("\n" + "="*55)
    print("--- AI-POWERED CHASE ANALYSIS ---")
    
    display_win_prob_chart(final_win_probability)
    
    print(f"Overall Required Rate: {required_rate_to_win:.2f} rpo.")
    print("="*55 + "\n")
    
    scaling_factor = runs_to_get / sum(over_weights) if sum(over_weights) > 0 else 1
    
    for i, over in enumerate(range(overs_completed + 1, total_overs + 1)):
        ai_adjusted_target = over_weights[i] * scaling_factor
        predicted_runs_this_over = over_weights[i]
        print(f"Over {over}: AI Target: {ai_adjusted_target:<5.2f}  |  AI Predicts: ~{predicted_runs_this_over:.2f} runs")
    
    # Print the final conclusion based on the simulation result
    print("-" * 55)
    if simulation_predicts_win:
        print(f"CONCLUSION: AI simulation finds a plausible path to VICTORY.")
    else:
        runs_short = target_score - simulated_score
        print(f"CONCLUSION: AI simulation predicts the team will NOT reach the target.")
        print(f"Falling short by approximately {runs_short:.0f} runs.")
    print("="*55)

if __name__ == "__main__":
    run_ai_prediction_calculator()