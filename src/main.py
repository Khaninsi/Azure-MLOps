import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--max_depth", required=False, default=15, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    ###################
    # <Load the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    
    df_bank = pd.read_csv(args.data)

    # Log the size of dataframe
    mlflow.log_metric("num_samples", df_bank.shape[0])
    mlflow.log_metric("num_features", df_bank.shape[1] - 1)
    
    ###################
    # </Load the data>
    ###################
    
    ##################
    #<Data preprocessing>
    ##################
    
    # Copying original dataframe
    df_bank_ready = df_bank.copy()
    
    ### Data manupulation
    # Clean belance < 0 to be 0
    df_bank.loc[df_bank['balance']<0, 'balance'] = 0
    
    # Select Features
    feature = df_bank.drop('deposit', axis=1)

    # Select Target
    target = df_bank['deposit'].apply(lambda deposit: 1 if deposit == 'yes' else 0)

    # Set Training and Testing Data
    X_train, X_test, y_train, y_test = train_test_split(feature , target, 
                                                        shuffle = True, 
                                                        test_size=0.2, 
                                                        random_state=1)

    # Transform data
    numeric_columns = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

    scaler = StandardScaler()
    one_hot_encoder = OneHotEncoder()

    preprocessor = ColumnTransformer(transformers=[
        ('num', scaler, numeric_columns),
        ('cat', one_hot_encoder, categorical_columns)
    ])

    # We fit preprocessor with X_train to prevent overfitting 
    preprocessor.fit(X_train)
    
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)


    print(f"Training with data of shape {X_train_preprocessed.shape}")
    
    ##################
    #</Data preprocessing>
    ##################

    ##################
    #<train the model>
    ##################
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_split=40, min_samples_leaf=60
    )
    clf.fit(X_train_preprocessed, y_train)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', clf)
    ])

    y_pred = pipeline.predict(X_test)

#     y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    ###################
    #</train the model>
    ###################

    ##########################
    #<save and register model>
    ##########################
    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
    ###########################
    #</save and register model>
    ###########################
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
