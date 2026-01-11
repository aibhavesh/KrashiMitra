from fusion_ai.crop_logic import recommend_crops

def fuse_all(soil_type, salinity, white_ratio, questionnaire):
    root_health = "healthy"

    if questionnaire["root_layer"] == "deep":
        root_health = "restricted"

    crops = recommend_crops(
        soil_type=soil_type,
        moisture=questionnaire["moisture"],
        salinity=salinity,
        season=questionnaire["season"]
    )

    return {
        "soil": soil_type,
        "salinity": salinity,
        "white_ratio": white_ratio,
        "root_condition": root_health,
        "recommended_crops": crops
    }
