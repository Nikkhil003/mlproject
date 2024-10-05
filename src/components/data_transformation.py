import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
# The @dataclass decorator in Python is used to automatically generate special methods for classes, such as __init__(), __repr__(), __eq__(), and others, based on the class attributes. This can significantly reduce boilerplate code when creating classes that are primarily used to store data.
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor_obj.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            # Define a pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    # Step 1: Handle missing values by imputing with the median value
                    ("imputer", SimpleImputer(strategy="median")),
                    # Step 2: Scale the numerical features using StandardScaler
                    ("scaler", StandardScaler())
                ]
            )

            # Define a pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    # Step 1: Handle missing values by imputing with the most frequent value
                    ("imputer", SimpleImputer(strategy="most_frequent")),

                    # Step 2: Convert categorical variables to one-hot encoded variables
                    ("one_hot_encoder", OneHotEncoder()),

                    # Step 3: Scale the categorical features (without centering)
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine the numerical and categorical pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    # Apply the numerical pipeline to the numerical columns
                    ("num_pipeline", num_pipeline, numerical_columns),
                    
                    # Apply the categorical pipeline to the categorical columns
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read the data successfully")

            logging.info("Obtaining the data transformer object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)