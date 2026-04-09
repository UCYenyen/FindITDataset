import json
import os

with open(r'c:\Rei\UC\FindIT\FindITDataset\Notebook\inference.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Add import shap
        if any('import joblib' in source for source in cell['source']):
            new_source = []
            for s in cell['source']:
                new_source.append(s)
                if 'import joblib' in s:
                    new_source.append('import shap\n')
            cell['source'] = new_source
        
        # Change OUTPUT_DIR
        new_source = []
        for s in cell['source']:
            s = s.replace(
                "pd.read_csv(os.path.join(OUTPUT_DIR, 'dataset_daily_processed.csv'))",
                "pd.read_csv(os.path.join('../test_data', 'dataset_daily_test.csv'))"
            )
            new_source.append(s)
        cell['source'] = new_source

        # Add SHAP explanation
        if any('INFERENCE PREVIEW' in source for source in cell['source']):
            # It currently has: display(df_clean[['Date', 'Forecast_MWh', 'Anomaly_Flag']].tail(10)) without trailing \n
            # Let's ensure a \n is appended
            if not cell['source'][-1].endswith('\n'):
                cell['source'][-1] += '\n'
            
            cell['source'].extend([
                "print('\\n--- EXPLAINABILITY (SHAP) ON LATEST INPUT ---')\n",
                "explainer = shap.TreeExplainer(lgbm_model)\n",
                "shap_vals = explainer.shap_values(df_clean[features])\n",
                "latest_shap = shap_vals[-1]\n",
                "feature_impacts = list(zip(features, latest_shap))\n",
                "feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)\n",
                "for i, (feat, impact) in enumerate(feature_impacts[:3]):\n",
                "    direction = 'Meningkatkan Prediksi (Positif)' if impact > 0 else 'Menurunkan Prediksi (Negatif)'\n",
                "    print(f'{i+1}. {feat} ➔ {direction} berdampak {abs(impact):,.2f} MWh.')"
            ])

with open(r'c:\Rei\UC\FindIT\FindITDataset\Notebook\inference.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
