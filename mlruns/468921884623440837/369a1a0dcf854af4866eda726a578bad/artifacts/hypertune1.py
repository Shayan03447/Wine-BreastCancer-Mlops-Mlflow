from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name='target')

# splitting into train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.2,random_state=42)

rf=RandomForestClassifier(random_state=42)

param_grid={
    'n_estimators':[10,50,100],
    'max_depth':[None, 10, 20, 30]
}

Grid_search=GridSearchCV(estimator=rf,param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

mlflow.set_experiment('Breast-Cancer-rf')
with mlflow.start_run() as parent:

    # Run without Mlflow
    Grid_search.fit(X_train, y_train)

    # Log all the child runs
    for i in range(len(Grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(Grid_search.cv_results_['params'][i])
            mlflow.log_metric('accuracy' ,Grid_search.cv_results_['mean_test_score'][i])


    # Display the best parameter and best score
    best_params=Grid_search.best_params_
    best_score=Grid_search.best_score_
    # Log params
    mlflow.log_params(best_params)
    # Log metrics
    mlflow.log_metric("accuracy" ,best_score)
    # Log training data
    train_df=X_train.copy()
    train_df['target']=y_train

    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "Training")

    # log test data
    test_df=X_test.copy()
    test_df['traget']=y_test

    test_df=mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "Testing")

    # Log the source code
    mlflow.log_artifact(__file__)

    # log the bst model
    mlflow.sklearn.log_model(Grid_search.best_estimator_, "random_forest")

    # Tags
    mlflow.set_tag("Author","Shayan")



    print(best_params)
    print(best_score)