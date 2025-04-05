# agents/market_researcher.py
import pandas as pd

class MarketResearcherAgent:
    def __init__(self, dataframe):
        self.df = dataframe

    def get_market_insight(self, crop_name):
        filtered = self.df[self.df['Product'].str.lower() == crop_name.lower()]
        
        if filtered.empty:
            return f"No market data found for crop: {crop_name}"

        avg_price = filtered['Market_Price_per_ton'].mean()
        avg_demand = filtered['Demand_Index'].mean()
        avg_supply = filtered['Supply_Index'].mean()
        avg_competitor_price = filtered['Competitor_Price_per_ton'].mean()
        avg_weather_impact = filtered['Weather_Impact_Score'].mean()
        trend = filtered['Seasonal_Factor'].mode().iloc[0] if not filtered['Seasonal_Factor'].mode().empty else "Unknown"

        return {
            "Average Price per Ton": round(avg_price, 2),
            "Average Demand Index": round(avg_demand, 2),
            "Average Supply Index": round(avg_supply, 2),
            "Average Competitor Price": round(avg_competitor_price, 2),
            "Average Weather Impact Score": round(avg_weather_impact, 2),
            "Most Common Seasonal Factor": trend
        }
