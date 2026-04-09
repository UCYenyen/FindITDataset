import json
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

raw_data_dir = os.path.join(PROJECT_ROOT, 'Raw Data')
output_dir = os.path.join(PROJECT_ROOT, 'Outputs')

print("Starting BMKG Weather integration...")

# 1. Parse BMKG Weather Data
# The BMKG repository mainly contains forecast data for specific days in JSON format.
# Let's extract the temperature and rainfall array from `31.71.01.1001.json` (Jakarta Pusat as a proxy for Indonesia's central weather or replacing noise)
bmkg_file = os.path.join(raw_data_dir, 'Kaggle_Climate', 'data-cuaca', '31.71.01.1001.json')

weather_records = []
try:
    with open(bmkg_file, 'r') as f:
        data = json.load(f)
        
    for area in data.get('data', []):
        for cuaca_day in area.get('cuaca', []):
            for hourly in cuaca_day:
                dt = pd.to_datetime(hourly['local_datetime']).date()
                temp = hourly['t']
                rain = hourly['tp']
                weather_records.append({'Date': dt, 'Temp': temp, 'Rain': rain})
                
    df_bmkg = pd.DataFrame(weather_records)
    # Aggregate to daily standard
    df_bmkg_daily = df_bmkg.groupby('Date').agg({'Temp': 'mean', 'Rain': 'sum'}).reset_index()
    df_bmkg_daily['Date'] = pd.to_datetime(df_bmkg_daily['Date'])
    print(f"BMKG Extracted {len(df_bmkg_daily)} days of real weather data from JSON.")
    
except Exception as e:
    print(f"Warning: BMKG JSON not available ({e}). Skipping BMKG integration.")
    df_bmkg_daily = pd.DataFrame(columns=['Date', 'Temp', 'Rain'])

# 2. Update the Daily Dataset
dataset_daily_path = os.path.join(output_dir, 'dataset_daily_processed.csv')
df_daily = pd.read_csv(dataset_daily_path)
df_daily['Date'] = pd.to_datetime(df_daily['Date'])

# Merge real BMKG Data into the dataset
# Because BMKG repo data might only contain forecast for specific future/current days (e.g. Oct 2025 in the JSON)
# For the purpose of historic backfilling (2018-2023), if real historic BMKG isn't fully in this JSON repo,
# We will overwrite matching dates, and for missing historic dates, use the statistical mean of the BMKG data extracted to anchor the synthetic data to reality.

if not df_bmkg_daily.empty:
    # Left merge to overwrite available dates
    df_daily = pd.merge(df_daily, df_bmkg_daily, on='Date', how='left')
    
    bmkg_matches = df_daily['Temp'].notna().sum()
    
    # Only replace dates where BMKG has real observations — keep existing
    # Kaggle Climate data for all other dates untouched
    df_daily['Avg_Temp'] = np.where(df_daily['Temp'].notna(), df_daily['Temp'], df_daily['Avg_Temp'])
    df_daily['Rainfall'] = np.where(df_daily['Rain'].notna(), df_daily['Rain'], df_daily['Rainfall'])
    
    df_daily.drop(columns=['Temp', 'Rain'], inplace=True)
    print(f"   Overwrote {bmkg_matches} dates with real BMKG observations, kept existing data for all others.")
else:
    print("No valid BMKG dates merged.")

# Save the updated daily dataset
df_daily.to_csv(dataset_daily_path, index=False)
print(f"✅ Real BMKG Weather Data integrated. Dataset updated: {dataset_daily_path}")


# 3. Update the Monthly Dataset Weather feature based on the new aggregated Daily Weather
dataset_monthly_path = os.path.join(output_dir, 'dataset_monthly_processed.csv')
df_monthly = pd.read_csv(dataset_monthly_path)

# Aggregate daily weather to monthly to update the monthly dataset
df_daily['Year'] = df_daily['Date'].dt.year
df_daily['Month'] = df_daily['Date'].dt.month
monthly_weather = df_daily.groupby(['Year', 'Month'])['Avg_Temp'].mean().reset_index()

df_monthly = pd.merge(df_monthly.drop(columns=['Avg_Temp']), monthly_weather, on=['Year', 'Month'], how='left')

# Reorder columns back to requested order
monthly_cols = ['Year', 'Month', 'Demand_GWh', 'GDP', 'Population', 'Industrial_Index', 'Avg_Temp', 'Lag_1', 'Lag_12', 'Rolling_12']
df_monthly = df_monthly[monthly_cols]

df_monthly.to_csv(dataset_monthly_path, index=False)
print(f"✅ Monthly Dataset updated with BMKG Weather aggregation: {dataset_monthly_path}")
