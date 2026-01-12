"""
Fusion Engine - Combines all AI analysis into final report
"""
from fusion_ai.crop_logic import recommend_crops


def fuse_all(soil_type, salinity, white_ratio, questionnaire):
    """
    Fuse all analysis data into a comprehensive soil health report
    
    Args:
        soil_type: Predicted soil type from CNN model
        salinity: Detected salinity level
        white_ratio: Ratio of white pixels in image
        questionnaire: Parsed questionnaire data
        
    Returns:
        dict: Complete analysis report with recommendations
    """
    # Analyze root health
    root_layer = questionnaire.get("root_layer", "shallow")
    if root_layer == "deep":
        root_health = "restricted"
    else:
        root_health = "healthy"
    
    # Get moisture info
    moisture = questionnaire.get("moisture", "Moist")
    season = questionnaire.get("season", "Kharif")
    
    # Generate crop recommendations
    crops = recommend_crops(
        soil_type=soil_type,
        moisture=moisture,
        salinity=salinity,
        season=season
    )
    
    # Build comprehensive report
    report = {
        "soil": soil_type.capitalize(),
        "salinity": salinity,
        "white_ratio": round(white_ratio, 4),
        "root_condition": root_health,
        "moisture_level": moisture,
        "season": season,
        "recommended_crops": crops,
        "analysis_date": None,  # Can be added by frontend
        "health_score": calculate_health_score(salinity, root_health, white_ratio)
    }
    
    return report


def calculate_health_score(salinity, root_health, white_ratio):
    """
    Calculate overall soil health score (0-100)
    
    Args:
        salinity: Salinity level
        root_health: Root condition
        white_ratio: White crust ratio
        
    Returns:
        int: Health score from 0-100
    """
    score = 100
    
    # Deduct for high salinity
    if salinity == "high":
        score -= 30
    elif salinity == "medium":
        score -= 15
    
    # Deduct for restricted roots
    if root_health == "restricted":
        score -= 20
    
    # Deduct for white crust
    score -= int(white_ratio * 100)
    
    return max(0, min(100, score))
