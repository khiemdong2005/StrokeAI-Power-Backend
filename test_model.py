import joblib
import pandas as pd
import numpy as np
import traceback

model = joblib.load("models/logistic_regression_pipeline.joblib")

def to01_str(x):
    return "1" if int(x) == 1 else "0"

df = pd.DataFrame([{
    "age": 25.0,
    "avg_glucose_level": 90.0,
    "bmi": 21.5,
    "log_avg_glucose_level": np.log1p(90.0),
    "log_bmi": np.log1p(21.5),

    # ✅ LOWERCASE đúng như categories_ trong model
    "gender": "male",
    "ever_married": "no",
    "work_type": "private",
    "Residence_type": "urban",
    "smoking_status": "never smoked",

    # ✅ 2 cột này model lưu category là '0','1' (string)
    "hypertension": to01_str(0),
    "heart_disease": to01_str(0),
}])

print("=== DF ===")
print(df)
print("\n=== DTYPES ===")
print(df.dtypes)

# ép toàn bộ cột categorical về object (chuỗi thường)
cat_cols = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
for c in cat_cols:
    df[c] = df[c].astype(object)

print("\n=== DTYPES after cast ===")
print(df.dtypes)

try:
    print("\n=== PRED ===")
    print(model.predict(df))

    print("\n=== PROBA ===")
    print(model.predict_proba(df))

except Exception as e:
    print("\n=== ERROR ===")
    print(str(e))
    print("\n=== TRACEBACK ===")
    print(traceback.format_exc())
