import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/fraud_model.joblib")

def predict_single(amount, hour, day_of_week, account_age_days):
    df = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "day_of_week": day_of_week,
        "account_age_days": account_age_days,
    }])
    prob = model.predict_proba(df)[0,1]
    label = "🚨 FRAUD" if prob >= 0.5 else "✅ LEGIT"
    return {"Risk Score": f"{prob*100:.2f}%", "Prediction": label}

def predict_batch(file):
    df = pd.read_csv(file.name)
    probs = model.predict_proba(df)[:,1]
    df["risk_score"] = (probs*100).round(2)
    df["prediction"] = np.where(probs>=0.5, "FRAUD", "LEGIT")
    return df

with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Simple Fraud Detection (XGBoost)")

    with gr.Tab("🔍 Single Transaction"):
        amount = gr.Number(label="Amount ($)", value=100.0)
        hour = gr.Slider(0, 23, 12, step=1, label="Hour of Day")
        day = gr.Slider(0, 6, 2, step=1, label="Day of Week (0=Mon)")
        age = gr.Number(label="Account Age (days)", value=365)
        btn = gr.Button("Predict")
        out = gr.JSON()
        btn.click(predict_single, [amount,hour,day,age], out)

    with gr.Tab("📊 Batch Upload"):
        file = gr.File(file_types=[".csv"])
        out_df = gr.Dataframe()
        file.change(predict_batch, file, out_df)

if __name__ == "__main__":
    demo.launch()
