import pandas as pd
import numpy as np

def generate_data(n=2000):
    np.random.seed(42)
    data = {
        "amount": np.random.lognormal(3, 1, n),
        "hour": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "account_age_days": np.random.randint(1, 2000, n),
    }
    df = pd.DataFrame(data)
    # Fraud label: higher prob for large amounts + night hours
    df["is_fraud"] = ((df["amount"] > 2000) & (df["hour"].isin([0,1,2,3,4]))) | (np.random.rand(n) < 0.05)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/transactions.csv", index=False)
    print("✅ Sample data generated:", df.shape)
