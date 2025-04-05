# app.py
from utils import preprocess
from agents.farmer_advisor import FarmerAdvisorAgent
from agents.market_researcher import MarketResearcherAgent

def main():
    print("🌾 Welcome to AI Smart Farming Advisor 🤖")
    print("Please enter the following field data:\n")

    try:
        soil_ph = float(input("📍 Soil pH (e.g., 6.5): "))
        soil_moisture = float(input("💧 Soil Moisture % (e.g., 45): "))
        temperature = float(input("🌡️ Temperature °C (e.g., 27): "))
        rainfall = float(input("☔ Rainfall in mm (e.g., 230): "))
    except ValueError:
        print("❌ Please enter valid numeric values.")
        return

    farmer_df = preprocess.load_farmer_data()
    market_df = preprocess.load_market_data()

    farmer_agent = FarmerAdvisorAgent(farmer_df)
    market_agent = MarketResearcherAgent(market_df)

    recommended_crop = farmer_agent.recommend_crop(
        soil_ph=soil_ph,
        soil_moisture=soil_moisture,
        temperature=temperature,
        rainfall=rainfall
    )

    print("\n🌾 Recommended Crop:")
    for crop in recommended_crop['Crop']:
        print(f" - {crop}")

        # Skip if no actual match
        if crop.lower().startswith("no crop"):
            continue

        market_data = market_agent.get_market_insight(crop)
        print(f"\n📊 Market Insight for {crop}:")

        if isinstance(market_data, dict):
            for key, value in market_data.items():
                print(f"  - {key}: {value}")
        else:
            print(f"  ⚠️ {market_data}")

if __name__ == "__main__":
    main()
