import sys 
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


from src.exception import CustomeException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_objects(self):

        try:
            numeric_features = ['reading score', 'writing score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns Standard-Scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numeric_features),
                ("cat_pipeline",cat_pipeline,categorical_features)
            ])

            return preprocessor
        
        except Exception as ex:

            raise CustomeException(ex,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data reading completed")

            logging.info("Obtaining Preprocessing Object")

            preprocessor_obj = self.get_data_transformer_objects()

            target_col_name="math score"

            input_feature_train_df = train_df.drop(columns=[target_col_name],axis=1)
            output_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name],axis=1)
            output_feature_test_df = test_df[target_col_name]

            logging.info("Applying the preprocessing object on training and testing input dataset")

            inp_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            inp_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                            (inp_feature_train_arr,np.array(output_feature_train_df))
            ]

            test_arr = np.c_[
                            (inp_feature_test_arr,np.array(output_feature_test_df))
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as ex:
            raise CustomeException(ex,sys)
        
