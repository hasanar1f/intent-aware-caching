from __future__ import annotations


UNKNOWN_INTENT = "unknown"

LABELS_30: tuple[str, ...] = (
    "general_explanations_definitions",
    "writing_creative",
    "math_problem_solving",
    "writing_business_marketing",
    "coding_generation",
    "how_to_guides",
    "relationship",
    "writing_academic",
    "health_medical_info",
    "summarization",
    "recommendations_products",
    "debugging_troubleshooting",
    "chitchat_smalltalk",
    "writing_editing_proofreading",
    "data_analysis_statistics",
    "entertainment_media",
    "career_jobs",
    "legal_info",
    "data_extraction_structuring",
    "translation",
    "personal_life_advice",
    "mental_wellbeing_support",
    "research_deep_dive",
    "philosophy_religion",
    "travel_planning",
    "education_study_help",
    "personal_finance",
    "food_cooking_recipes",
    "current_events_politics",
    "system_admin_devops",
)

MAP_30_TO_20: dict[str, str] = {
    "general_explanations_definitions": "general_explanations_definitions",
    "writing_creative": "writing_creation",
    "math_problem_solving": "education_learning",
    "writing_business_marketing": "writing_creation",
    "coding_generation": "coding_and_systems",
    "how_to_guides": "how_to_guides",
    "relationship": "advice_support",
    "writing_academic": "writing_creation",
    "health_medical_info": "health_medical_info",
    "summarization": "summarization",
    "recommendations_products": "recommendations",
    "debugging_troubleshooting": "coding_and_systems",
    "chitchat_smalltalk": "chitchat_smalltalk",
    "writing_editing_proofreading": "writing_editing_proofreading",
    "data_analysis_statistics": "research_deep_dive",
    "entertainment_media": "recommendations",
    "career_jobs": "career_jobs",
    "legal_info": "legal_info",
    "data_extraction_structuring": "research_deep_dive",
    "translation": "translation",
    "personal_life_advice": "advice_support",
    "mental_wellbeing_support": "advice_support",
    "research_deep_dive": "research_deep_dive",
    "philosophy_religion": "philosophy_religion",
    "travel_planning": "travel_planning",
    "education_study_help": "education_learning",
    "personal_finance": "personal_finance",
    "food_cooking_recipes": "food_cooking_recipes",
    "current_events_politics": "current_events_politics",
    "system_admin_devops": "coding_and_systems",
}

MAP_30_TO_10: dict[str, str] = {
    "general_explanations_definitions": "research_news",
    "writing_creative": "writing_language",
    "math_problem_solving": "career_education",
    "writing_business_marketing": "writing_language",
    "coding_generation": "coding_debugging",
    "how_to_guides": "how_to_practical",
    "relationship": "advice_support",
    "writing_academic": "writing_language",
    "health_medical_info": "high_stakes_info",
    "summarization": "writing_language",
    "recommendations_products": "recommendations_planning",
    "debugging_troubleshooting": "coding_debugging",
    "chitchat_smalltalk": "chitchat_smalltalk",
    "writing_editing_proofreading": "writing_language",
    "data_analysis_statistics": "data_work",
    "entertainment_media": "recommendations_planning",
    "career_jobs": "career_education",
    "legal_info": "high_stakes_info",
    "data_extraction_structuring": "data_work",
    "translation": "writing_language",
    "personal_life_advice": "advice_support",
    "mental_wellbeing_support": "advice_support",
    "research_deep_dive": "research_news",
    "philosophy_religion": "research_news",
    "travel_planning": "recommendations_planning",
    "education_study_help": "career_education",
    "personal_finance": "high_stakes_info",
    "food_cooking_recipes": "how_to_practical",
    "current_events_politics": "research_news",
    "system_admin_devops": "coding_debugging",
}


def labels_for(num_classes: int) -> tuple[list[str], dict[str, str] | None]:
    if num_classes == 30:
        return list(LABELS_30), None
    if num_classes == 20:
        label_map = MAP_30_TO_20
    elif num_classes == 10:
        label_map = MAP_30_TO_10
    else:
        raise ValueError(f"num_classes must be 10, 20 or 30, got {num_classes}")
    return list(dict.fromkeys(label_map[label] for label in LABELS_30)), label_map
