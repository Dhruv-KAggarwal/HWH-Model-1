import pandas as pd
import random

# Define possible values
skin_types = ["Oily", "Dry", "Normal", "Combination"]
hair_types = ["Straight", "Wavy", "Curly", "Frizzy"]
hair_thickness = ["Thin", "Medium", "Thick"]
risk_levels = ["Low", "Medium", "High"]
yes_no = ["Yes", "No"]

# Chemicals to avoid (randomly assigned)
chemicals_list = [
    "Sulfates, Parabens, Alcohol",
    "Fragrance, Silicones, PEGs",
    "Artificial Dyes, Formaldehyde",
    "Petroleum, Mineral Oil, Phthalates"
]

# Recommended ingredients for skincare and haircare
skincare_ingredients_list = [
    "Hyaluronic Acid, Vitamin C",
    "Niacinamide, Retinol",
    "Aloe Vera, Green Tea Extract",
    "Collagen, Ceramides"
]

haircare_ingredients_list = [
    "Argan Oil, Biotin",
    "Coconut Oil, Keratin",
    "Castor Oil, Jojoba Oil",
    "Aloe Vera, Shea Butter"
]

# Sample product names
skincare_products = ["Moisturizer A", "Serum B", "Sunscreen C", "Face Wash D"]
haircare_products = ["Shampoo X", "Conditioner Y", "Hair Oil Z", "Scalp Treatment W"]
supplements = ["Vitamin D", "Biotin", "Iron", "Collagen"]

# Generate 1000 rows of synthetic data
data = []
for i in range(1, 1001):
    row = {
        "ID": i,
        "Age": random.randint(18, 60),
        "Skin_Type": random.choice(skin_types),
        "Hair_Type": random.choice(hair_types),
        "Hair_Thickness": random.choice(hair_thickness),
        "Hair_Loss_Risk": random.choice(risk_levels),
        "Acne_Risk": random.choice(risk_levels),
        "Wrinkle_Risk": random.choice(risk_levels),
        "Vitamin_D_Deficiency": random.choice(yes_no),
        "Iron_Deficiency": random.choice(yes_no),
        "Lactose_Intolerance": random.choice(yes_no),
        "Allergy_Risk_Fragrance": random.choice(yes_no),
        "Allergy_Risk_Sulfates": random.choice(yes_no),
        "Allergy_Risk_Parabens": random.choice(yes_no),
        "Chemicals_to_Avoid": random.choice(chemicals_list),
        "UV_Sensitivity": random.choice(risk_levels),
        "Best_Skincare_Product": random.choice(skincare_products),
        "Skincare_Ingredients": random.choice(skincare_ingredients_list),
        "Best_Haircare_Product": random.choice(haircare_products),
        "Haircare_Ingredients": random.choice(haircare_ingredients_list),
        "Best_Supplement": random.choice(supplements)
    }
    data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("synthetic_dna_beauty_dataset.csv", index=False)

print("Dataset created successfully!")
