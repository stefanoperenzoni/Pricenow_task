import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor

from zenml.steps import step, Output
from zenml.pipelines import pipeline

@step
def data_loader() -> Output(
    data=pd.DataFrame
):
    album_pdf = pd.read_csv("../data/pitchfork.csv.gz")

    return album_pdf


@step
def data_pre_processing(album_pdf: pd.DataFrame) -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray
):
    random_seed = 123
    features = ['releaseyear', 'danceability', 'energy', 'key', 'loudness',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'Electronic_onehot', 'Experimental_onehot',
       'Folk/Country_onehot', 'Global_onehot', 'Jazz_onehot', 'Metal_onehot',
       'Other_onehot', 'Pop/R&B_onehot', 'Rap_onehot', 'Rock_onehot']

    #Filling empty valuesa and replacing 'none'
    album_pdf = album_pdf.fillna(value='Other')
    album_pdf['genre'] = album_pdf['genre'].replace('none', 'Other')

    # Used pd.get_dummies or scikit-learn OneHotEncoder
    one_hot_pdf = pd.get_dummies(album_pdf['genre']).add_suffix('_onehot')
    pre_album_pdf = album_pdf.drop('genre', axis=1)

    #Join the one_hot df
    pre_album_pdf = pre_album_pdf.join(one_hot_pdf) 

    #Create X datasets of indipendent variables and y with the score target
    X = pre_album_pdf[features].to_numpy()
    y = album_pdf['score'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = random_seed)

    return X_train, X_test, y_train, y_test

@step
def gradient_boosting_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RegressorMixin:
    random_seed = 123
    
    #GradientBoostingRegressor
    gr_boost_model = GradientBoostingRegressor(random_state=random_seed)
    gr_boost_model.fit(X_train, y_train)
    return gr_boost_model


@pipeline
def model_pipeline(step_1, step_2, step_3):
    album_pdf = step_1()
    X_train, X_test, y_train, y_test = step_2(album_pdf)
    step_3(X_train, y_train)


if __name__ == "__main__":

    model_pipeline_instance = model_pipeline(
    step_1=data_loader(),
    step_2=data_pre_processing(),
    step_3=gradient_boosting_trainer()
    )

    model_pipeline_instance.run(run_name="GB_Pipeline_Run_{time}")