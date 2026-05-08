import json
import os
import time
import numpy as np

PROFILES_FILE = "profiles.json"

def calculate_similarity(sig1, sig2):
    """Calculates similarity, handles mismatched signature lengths safely."""
    s1 = np.array(sig1)
    s2 = np.array(sig2)
    if s1.shape != s2.shape:
        return 999.0 # Incompatible signature format
    return float(np.linalg.norm(s1 - s2))

def load_all_profiles():
    if os.path.exists(PROFILES_FILE) and os.path.getsize(PROFILES_FILE) > 0:
        try:
            with open(PROFILES_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: {PROFILES_FILE} is corrupt. Initializing new profile database.")
            return {}
    return {}

def save_all_profiles(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=4)

def identify_user(current_signature):
    """Attempts to find a matching face signature in the database."""
    if current_signature is None:
        return None, None
        
    profiles = load_all_profiles()
    # Tightened threshold for 8-dimension signature
    # Since we have more points, we expect a slightly higher numerical distance for the same 'closeness'
    # But 0.10 is quite strict for 8 ratios.
    min_dist = 0.12 
    recognized_name = None

    print(f"DEBUG: Checking signature length {len(current_signature)}")
    for name, data in profiles.items():
        if not data: continue 
        
        sig = data.get("face_signature")
        if sig and isinstance(sig, list):
            # Compatibility check: only compare if lengths match
            if len(sig) == len(current_signature):
                dist = calculate_similarity(current_signature, sig)
                print(f"DEBUG: Distance to {name} = {dist:.4f}")
                if dist < min_dist:
                    min_dist = dist
                    recognized_name = name
            else:
                print(f"DEBUG: Skipping {name} due to signature length mismatch ({len(sig)} vs {len(current_signature)})")
    
    if recognized_name:
        print(f"DEBUG: Recognized as {recognized_name}")
    else:
        print("DEBUG: No match found within threshold.")

    return recognized_name, profiles.get(recognized_name) if recognized_name else None

def register_user(name, signature, skin_data, user_info=None):
    """Adds a new user or updates an existing one via Web UI."""
    profiles = load_all_profiles()
    name = name.strip()
    
    if name not in profiles:
        print(f"\n[NEW] Registering New Profile: {name}")
        # user_info expected: {"age": ..., "concerns": ..., "budget": ..., "password": ...}
        profiles[name] = {
            "user_info": {
                "name": name, 
                "age": user_info.get('age', 'N/A') if user_info else 'N/A', 
                "concerns": user_info.get('concerns', 'General') if user_info else 'General', 
                "budget": user_info.get('budget', 'Mid') if user_info else 'Mid',
                "password": user_info.get('password', '') if user_info else ''
            },
            "history": []
        }
    else:
        print(f"\n[UPDATE] Updating Existing Profile: {name}")
        if user_info and 'password' in user_info:
            profiles[name]["user_info"]["password"] = user_info['password']
    
    return update_user_scan(name, signature, skin_data, profiles)

def update_user_scan(name, signature, skin_data, profiles=None):
    """Updates the latest scan data for a user."""
    name = name.strip()
    if profiles is None:
        profiles = load_all_profiles()
        
    if name not in profiles:
        return None

    profiles[name]["face_signature"] = signature
    profiles[name]["latest_scan"] = {
        "timestamp": time.ctime(),
        "interpretation": skin_data['interpretation'],
        "raw_data": skin_data['regions']
    }

    save_all_profiles(profiles)
    return profiles[name]
