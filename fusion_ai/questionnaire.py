def analyze_questionnaire(ans):
    score = {}

    # Moisture score
    if ans["water_stay"] == "Long time":
        score["drainage"] = "Poor"
    elif ans["water_stay"] == "Some time":
        score["drainage"] = "Medium"
    else:
        score["drainage"] = "Good"

    # Texture
    if ans["feel"] == "Sticky" or ans["cracks"] == "Deep":
        score["texture"] = "Clay"
    elif ans["feel"] == "Loose":
        score["texture"] = "Sandy"
    else:
        score["texture"] = "Loamy"

    # Stress
    stress = 0
    if ans["yield"] == "Low": stress += 2
    if ans["white_crust"] == "Yes": stress += 2
    if ans["moisture"] == "Very wet": stress += 1
    score["stress"] = stress

    return score

def parse_questionnaire(q):
    return {
        "season": q["season"],
        "crop": q["crop"],
        "moisture": q["moisture"],
        "texture": q["texture"],
        "cracks": q["cracks"],
        "absorption": q["absorption"],
        "crust": q["crust"],
        "root_layer": q["root_layer"],
        "yield": q["yield"]
    }