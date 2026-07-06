# validation_report.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandera import check_output

@check_output(OceanDataSchema, lazy=True)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

if __name__ == "__main__":
    df = load_data("data/processed/historical_features.parquet")
    
    # Generate quality report
    report = {
        "total_records": len(df),
        "stations_processed": df["station_id"].nunique(),
        "completeness": (1 - df.isnull().mean()).to_dict(),
        "validation_pass_rate": (df["validation_status"] != "failed").mean()
    }
    
    # Plot key distributions
    fig, ax = plt.subplots(3, 2, figsize=(15, 10))
    sns.histplot(df["significant_wave_height"], ax=ax[0, 0], kde=True)
    sns.histplot(df["total_energy"], ax=ax[0, 1], kde=True)
    sns.scatterplot(x="total_energy", y="significant_wave_height", data=df, ax=ax[1, 0])
    sns.histplot(df["bimodality"], ax=ax[1, 1], kde=True)
    sns.scatterplot(x="wind_speed", y="wave_wind_alignment", data=df, ax=ax[2, 0])
    plt.tight_layout()
    plt.savefig("validation_report.png")
    
    print("Validation Report:")
    print(json.dumps(report, indent=2))