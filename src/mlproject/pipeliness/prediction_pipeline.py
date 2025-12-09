import os
import sys
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.utils import load_object

class PredictPipeline:
    def predict(self, features):
        try:
            project_root = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../../"
                )
            )

            model_path = os.path.join(project_root, "artifacts", "model.pkl")
            preprocessor_path = os.path.join(project_root, "artifacts", "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            transformed_data = preprocessor.transform(features)
            prediction = model.predict(transformed_data)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)