import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import mlflow

from zenml.integrations.mlflow.steps import mlflow_deployer_step
from zenml.services.utils import load_last_service_from_step
from zenml.integrations.mlflow.steps import MLFlowDeployerParameters
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from zenml.steps import step, Output, BaseParameters
from zenml.pipelines import pipeline

class DataProcessingParam(BaseParameters):
    random_seed = 123
    test_size = 0.3
    
    
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
    model_name: str = "model"


class DeploymentTriggerParameters(BaseParameters):
    """Parameters that are used to trigger the deployment."""
    min_R2: float=0


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters.
    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    pipeline_step_name: str
    running: bool = True
    model_name: str = "model"

@step(enable_cache=False)
def data_loader() -> Output(
    data=pd.DataFrame
):
    '''
        Load data from the path ../data/pitchfork.csv.gz
    '''
    album_pdf = pd.read_csv("../data/pitchfork.csv.gz")

    return album_pdf


@step(enable_cache=False)
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

    print(y_train.shape)

    return X_train, X_test, y_train, y_test


def feat_engineering(X, feature_list):
    # Used pd.get_dummies or scikit-learn OneHotEncoder
    one_hot_pdf = pd.get_dummies(X['genre']).add_suffix('_onehot')
    X = X.drop('genre', axis=1)
    #Join the one_hot df
    X_feat = X.join(one_hot_pdf)

    return X_feat[feature_list]

@step(enable_cache=False)
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
        enable_cache=False,
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
        enable_cache=False,
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


@step(enable_cache=False)
def deployment_trigger(
    metrics: dict,
    params: DeploymentTriggerParameters,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model R2 and decides if it is good enough to deploy."""
    #For the example, I used a R2 of 0. Normally, this is not a good condition
    return metrics['R2'] > params.min_R2

#Step
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


#Inference Pipeline TODO
@step(enable_cache=False)
def data_batch_loader() -> Output(
    data=pd.DataFrame
):
    '''
        Load data from the path ../data/inference.csv
    '''
    album_pdf = pd.read_csv("../data/inference.csv")

    #Filling empty valuesa and replacing 'none'
    album_pdf = album_pdf.fillna(value='Other')
    album_pdf['genre'] = album_pdf['genre'].replace('none', 'Other')

    return album_pdf.drop('score', axis=1)


@step(enable_cache=False)
def inference_feature_engineering(X: pd.DataFrame, params: FeatureEngineeringParam) -> Output(
    X=pd.DataFrame
):
    '''
    Apply the same feature engineering steps to X_train and X_test
    '''
    X = feat_engineering(X, params.features)

    return X


@step(enable_cache=False)
def model_service_loader(
    params: MLFlowDeploymentLoaderStepParameters,
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline."""
    # get the MLflow model deployer stack component

    model_deployer_l = MLFlowModelDeployer.get_active_model_deployer()
    print(model_deployer_l)

    # fetch existing services with same pipeline name, step name and model name
    existing_service = model_deployer_l.find_model_server(
        pipeline_name=params.pipeline_name,
        pipeline_step_name=params.pipeline_step_name,
        model_name=params.model_name,
        running=params.running,
    )

    if not existing_service:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{params.pipeline_step_name} step in the {params.pipeline_name} "
            f"pipeline for the '{params.model_name}' model is currently "
            f"running."
        )

    return existing_service[0]


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> Output(predictions=np.ndarray):
    """Run a inference request against a prediction service."""
    service.start(timeout=60)  # should be a NOP if already started
    prediction = service.predict(data)
    prediction = prediction.argmax(axis=-1)
    print("Prediction: ", prediction)
    return prediction

@pipeline
def inference_pipeline(
    data_batch_loader,
    inference_feature_eng,
    model_service_loader,
    predictor,
):
    batch_data = data_batch_loader()
    inference_data = inference_feature_eng(batch_data)
    model_deployment_service = model_service_loader()
    predictor(model_deployment_service, inference_data)



def main(config = 'deploy'):

    deploy = config == 'deploy' or config == 'deploypredict'
    predict = config == 'predict' or config == 'deploypredict'

    if deploy:
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

        model_pipeline_instance.run(config_path="config.yml")

    if predict: 
        inference = inference_pipeline(
        data_batch_loader=data_batch_loader(),
        inference_feature_eng=inference_feature_engineering(),
        model_service_loader=model_service_loader(
                MLFlowDeploymentLoaderStepParameters(
                    pipeline_name="model_pipeline",
                    pipeline_step_name="model_deployer"                
                )
            ),
            predictor=predictor(),
        )

        inference.run(run_name="Inference_Pipeline_Run_{time}")



if __name__ == "__main__":
    type_pipeline = sys.argv[1]
    main('deploy')