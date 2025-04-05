from langchain_core.runnables import RunnableLambda
from utils import preprocess
from agents.farmer_advisor import FarmerAdvisorAgent
from agents.market_researcher import MarketResearcherAgent
import numpy as np
from langchain_core.runnables import RunnableLambda

def safe_cast(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    return value

def advisor_logic(inputs):
    try:
        soil_ph = inputs["Soil_pH"]
        soil_moisture = inputs["Soil_Moisture"]
        temperature = inputs["Temperature_C"]
        rainfall = inputs["Rainfall_mm"]
    except KeyError as e:
        return {"error": f"❌ Missing input: {e}"}

    farmer_df = preprocess.load_farmer_data()
    market_df = preprocess.load_market_data()

    farmer_agent = FarmerAdvisorAgent(farmer_df)
    market_agent = MarketResearcherAgent(market_df)

    recommended_crop_df = farmer_agent.recommend_crop(
        soil_ph=soil_ph,
        soil_moisture=soil_moisture,
        temperature=temperature,
        rainfall=rainfall
    )

    output = {}

    for crop in recommended_crop_df['Crop']:
        crop = crop.strip()
        output[crop] = {}

        if crop.lower().startswith("no crop"):
            output[crop] = "No suitable crop found."
            continue

        market_data = market_agent.get_market_insight(crop)

        if isinstance(market_data, dict):
            output[crop] = {k: safe_cast(v) for k, v in market_data.items()}
        else:
            output[crop] = {"note": str(market_data)}

    return output
smart_farming_chain = RunnableLambda(advisor_logic)