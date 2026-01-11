def recommend_crops(soil_type, moisture, salinity, season):
    if salinity == "high":
        return ["Barley", "Cotton"]

    if soil_type == "loamy" and moisture == "moist":
        return ["Maize", "Wheat", "Pulses"]

    if soil_type == "sandy":
        return ["Millet", "Groundnut"]

    return ["Sorghum", "Maize"]
