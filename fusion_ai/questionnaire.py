"""
Questionnaire Parser - Process farmer responses about soil conditions
"""

def analyze_questionnaire(ans):
    """
    Analyze questionnaire answers to extract soil health indicators
    
    Args:
        ans: Dictionary of answers
        
    Returns:
        dict: Analyzed scores and classifications
    """
    score = {}

    # Moisture/Drainage score
    water_stay = ans.get("water_stay", "")
    if water_stay == "Long time":
        score["drainage"] = "Poor"
    elif water_stay == "Some time":
        score["drainage"] = "Medium"
    else:
        score["drainage"] = "Good"

    # Texture analysis
    feel = ans.get("feel", "")
    cracks = ans.get("cracks", "")
    
    if feel == "Sticky" or cracks == "Deep":
        score["texture"] = "Clay"
    elif feel == "Loose":
        score["texture"] = "Sandy"
    else:
        score["texture"] = "Loamy"

    # Stress indicators
    stress = 0
    if ans.get("yield") == "Low":
        stress += 2
    if ans.get("white_crust") == "Yes" or ans.get("crust") == "Yes":
        stress += 2
    if ans.get("moisture") == "Very wet":
        stress += 1
    score["stress"] = stress

    return score


def parse_questionnaire(q):
    """
    Parse and validate questionnaire data from frontend
    
    Args:
        q: Raw questionnaire dictionary
        
    Returns:
        dict: Cleaned and validated questionnaire data
    """
    # Provide defaults for missing keys
    return {
        "season": q.get("season", "Kharif"),
        "crop": q.get("crop", "Unknown"),
        "moisture": q.get("moisture", "Moist"),
        "texture": q.get("texture", "Loamy"),
        "cracks": q.get("cracks", "None"),
        "absorption": q.get("absorption", "Fast"),
        "crust": q.get("crust", "No"),
        "root_layer": q.get("root_layer", "shallow"),
        "yield": q.get("yield", "Average"),
        "water_stay": q.get("water_stay", "Drains fast"),
        "feel": q.get("feel", "Soft")
    }