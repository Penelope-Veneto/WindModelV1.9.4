#/src/inference.py
import joblib

class WindInference:
    def __init__(self, model_path):
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.features = payload["features"]

    def predict(self, df):
        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features at inference: {missing}")

        x = df.tail(1)[self.features]
        pred = self.model.predict(x)[0]
        return max(0, pred)