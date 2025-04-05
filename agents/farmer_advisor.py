# agents/farmer_advisor.py

class FarmerAdvisorAgent:
    def __init__(self, dataset):
        self.df = dataset

    def recommend_crop(self, soil_ph, soil_moisture, temperature, rainfall):
        filtered_df = self.df[
            (self.df['Soil_pH'].between(soil_ph - 0.5, soil_ph + 0.5)) &
            (self.df['Soil_Moisture'].between(soil_moisture - 5, soil_moisture + 5)) &
            (self.df['Temperature_C'].between(temperature - 2, temperature + 2)) &
            (self.df['Rainfall_mm'].between(rainfall - 50, rainfall + 50))
        ]

        if filtered_df.empty:
            return {"Crop": ["No crop match found for the given conditions."]}

        best_crop = (
            filtered_df.groupby("Crop_Type")["Crop_Yield_ton"]
            .mean()
            .idxmax()
        )

        return {"Crop": [best_crop]}
