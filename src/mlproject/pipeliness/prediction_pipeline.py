import sys
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.utils import load_object

class PredictPipeline:
    def predict(self, features: pd.DataFrame):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_transformed = preprocessor.transform(features)

            prediction = model.predict(data_transformed)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)