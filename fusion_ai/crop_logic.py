"""
Crop Recommendation Logic - Maps soil conditions to suitable crops
"""

def recommend_crops(soil_type, moisture, salinity, season):
    """
    Recommend suitable crops based on soil conditions
    
    Args:
        soil_type: Type of soil (sandy, loamy, clay)
        moisture: Moisture level (dry, moist, wet)
        salinity: Salinity level (low, medium, high)
        season: Growing season (Rabi, Kharif, Zaid)
        
    Returns:
        list: List of recommended crop names
    """
    # Normalize inputs
    soil_type = soil_type.lower() if soil_type else "loamy"
    moisture = moisture.lower() if moisture else "moist"
    salinity = salinity.lower() if salinity else "low"
    season = season.capitalize() if season else "Kharif"
    
    crops = []
    
    # High salinity requires salt-tolerant crops
    if salinity == "high":
        crops = ["Barley", "Cotton", "Date Palm"]
        return crops
    
    # Medium salinity - moderately tolerant crops
    if salinity == "medium":
        crops = ["Wheat", "Cotton", "Barley", "Sorghum"]
        return crops
    
    # Low salinity - based on soil type and moisture
    
    # Loamy soil (best for most crops)
    if soil_type == "loamy":
        if season == "Rabi":  # Winter crops
            crops = ["Wheat", "Barley", "Gram", "Mustard", "Peas"]
        elif season == "Kharif":  # Monsoon crops
            crops = ["Rice", "Maize", "Cotton", "Soybean", "Groundnut"]
        else:  # Zaid (summer)
            crops = ["Watermelon", "Cucumber", "Muskmelon", "Vegetables"]
    
    # Sandy soil (good drainage, low nutrients)
    elif soil_type == "sandy":
        if moisture in ["dry", "moist"]:
            crops = ["Millet", "Groundnut", "Watermelon", "Pulses"]
        else:
            crops = ["Millet", "Groundnut", "Bajra"]
    
    # Clay soil (heavy, good for water-intensive crops)
    elif soil_type == "clay":
        if moisture in ["wet", "very wet"]:
            crops = ["Rice", "Sugarcane", "Wheat", "Cotton"]
        else:
            crops = ["Cotton", "Wheat", "Sorghum", "Sunflower"]
    
    # Default fallback
    if not crops:
        crops = ["Maize", "Sorghum", "Pulses"]
    
    return crops[:5]  # Return top 5 recommendations
