import os
import sys
import pandas as pd
import joblib

from src.exception import CustomException


# ---------- UTILITY FUNCTION ----------
def load_object(file_path: str):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)


# ---------- PREDICTION PIPELINE ----------
class PredictPipeline:
    def __init__(self):
        try:
            # Absolute path to project root
            self.base_dir = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )

            self.model_path = os.path.join(self.base_dir, "artifacts", "model.pkl")
            self.preprocessor_path = os.path.join(
                self.base_dir, "artifacts", "preprocessor.pkl"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            print("Before loading model and preprocessor")

            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            print("After loading model and preprocessor")

            # Transform input features
            data_scaled = preprocessor.transform(features)

            # Predict
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


# ---------- CUSTOM DATA ----------
class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Convert user input into a pandas DataFrame
        EXACTLY matching training columns
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [
                    self.parental_level_of_education
                ],
                "lunch": [self.lunch],
                "test_preparation_course": [
                    self.test_preparation_course
                ],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            df = pd.DataFrame(custom_data_input_dict)
            print("Custom input DataFrame:\n", df)

            return df

        except Exception as e:
            raise CustomException(e, sys)
