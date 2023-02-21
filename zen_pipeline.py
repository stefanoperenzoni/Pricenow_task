import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import mlflow

from zenml.integrations.mlflow.steps import mlflow_deployer_step
from zenml.integrations.mlflow.steps import MLFlowDeployerParameters
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step


from zenml.steps import step, Output, BaseParameters
from zenml.pipelines import pipeline

class DataProcessingParam(BaseParameters):
    random_seed = 123
    
    
class FeatureEngineeringParam(BaseParameters):
    features = ['releaseyear', 'danceability', 'energy', 'key', 'loudness',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'Electronic_onehot', 'Experimental_onehot',
       'Folk/Country_onehot', 'Global_onehot', 'Jazz_onehot', 'Metal_onehot',
       'Other_onehot', 'Pop/R&B_onehot', 'Rap_onehot', 'Rock_onehot']
    

class ModelingParam(BaseParameters):
    model_parameters = {
        'random_state' : 123,
        'n_estimators' : 300,
        'learning_rate' : 0.01
    }
    model_name = "Gradient_Boosted_model"


class DeploymentTriggerParameters(BaseParameters):
    """Parameters that are used to trigger the deployment."""

    min_R2: float=0

@step
def data_loader() -> Output(
    data=pd.DataFrame
):
    '''
        Load data from the path ../data/pitchfork.csv.gz
    '''
    album_pdf = pd.read_csv("../data/pitchfork.csv.gz")

    return album_pdf


@step
def data_pre_processing(album_pdf: pd.DataFrame, params: DataProcessingParam) -> Output(
    X_train=pd.DataFrame, X_test=pd.DataFrame, y_train=pd.Series, y_test=pd.Series
):
    '''
    Preprocess the loaded data to replace the nan values. Split the dataset into test and training with test_size = 0.3
    '''
    #Filling empty valuesa and replacing 'none'
    album_pdf = album_pdf.fillna(value='Other')
    album_pdf['genre'] = album_pdf['genre'].replace('none', 'Other')

    #Create X datasets of indipendent variables and y with the score target
    X = album_pdf.drop('score', axis=1)
    y = album_pdf['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = params.random_seed)

    return X_train, X_test, y_train, y_test


def feat_engineering(X, feature_list):
    # Used pd.get_dummies or scikit-learn OneHotEncoder
    one_hot_pdf = pd.get_dummies(X['genre']).add_suffix('_onehot')
    X = X.drop('genre', axis=1)
    #Join the one_hot df
    X_feat = X.join(one_hot_pdf)

    return X_feat[feature_list]

@step
def feature_engineering(X_train: pd.DataFrame, X_test: pd.DataFrame, params: FeatureEngineeringParam) -> Output(
    X_train=pd.DataFrame, X_test=pd.DataFrame
):
    '''
    Apply the same feature engineering steps to X_train and X_test
    '''
    X_train = feat_engineering(X_train, params.features)

    X_test = feat_engineering(X_test, params.features)



    return X_train, X_test

@step(
        experiment_tracker = "mlflow_experiment_tracker",
)
def gradient_boosting_trainer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: ModelingParam
) -> RegressorMixin:
    '''
    Train a gradient boosted regressor, logs it using MLFlow
    '''
    
    #GradientBoostingRegressor
    gr_boost_model = GradientBoostingRegressor(**params.model_parameters)

    gr_boost_model.fit(X_train, y_train)

    #MLFlow logging
    for param in params.model_parameters.keys():
        mlflow.log_param(param, params.model_parameters[param])
    mlflow.sklearn.log_model(gr_boost_model, params.model_name)

    return gr_boost_model


@step(
        experiment_tracker = "mlflow_experiment_tracker",
)
def model_evaluation(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> dict:
    '''
    Evaluate the given model on a test set, return R2 metric
    '''
    pred = model.predict(X_test)
    R2_metric = r2_score(y_test, pred)

    mlflow.log_metric('R2', R2_metric)

    return_dict = {'R2': R2_metric}

    return return_dict


@step
def deployment_trigger(
    metrics: dict,
    params: DeploymentTriggerParameters,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model R2 and decides if it is good enough to deploy."""
    #For the example, I used a R2 of 0. Normally, this is not a good condition
    return metrics['R2'] > params.min_R2


model_deployer = mlflow_model_deployer_step


@pipeline
def model_pipeline(data_load_step, preprocess_step, feature_engineering, train_step, evaluate_step, deployment_trigger, model_deployer):
    album_pdf = data_load_step()
    X_train, X_test, y_train, y_test = preprocess_step(album_pdf)
    X_train, X_test = feature_engineering(X_train, X_test)
    model = train_step(X_train, y_train)
    evaluation_metrics = evaluate_step(model, X_test, y_test)
    deployment_decision = deployment_trigger(evaluation_metrics)
    model_deployer(deployment_decision, model)


#Deployment
@pipeline
def inference_pipeline(
    data_batch_loader,
    preprocess_step,
    model_service_loader,
    predictor,
):
    '''
    TODO:
        Did not implemented because of time concerns

        A inference pipeline to load the model and make it available for batch inference

        Necessary steps are:
        - loading the batch data
        - Preprocessing the data
        - Load the deployed model to make predictions (Deployed model generated using MlFlow Deployment Step)
        - Call the infence step (predictor) with input the service (Deployed model) and the loaded data
        - Return the made batch predictions
    '''
    pass


if __name__ == "__main__":

    model_pipeline_instance = model_pipeline(
    data_load_step = data_loader(),
    preprocess_step = data_pre_processing(),
    feature_engineering = feature_engineering(),
    train_step = gradient_boosting_trainer(),
    evaluate_step = model_evaluation(),
    deployment_trigger = deployment_trigger(),
    model_deployer=model_deployer(
        params=MLFlowDeployerParameters(timeout=60)
    )
    )

    model_pipeline_instance.run(run_name="GB_Pipeline_Run_{time}", config_path="config.yml")