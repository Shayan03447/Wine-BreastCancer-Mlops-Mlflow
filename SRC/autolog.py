import mlflow
import mlflow.sklearn
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from  sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load the DataSet
wine=load_wine()
X=wine.data
y=wine.target

# Train test split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.10, random_state=42)

# Define the parameters for the RF model
max_depth=10
n_estimators=5


# Mention your experiment below
mlflow.autolog()
mlflow.set_experiment('MLFLOW_Experiment1')

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)

   
    #mlflow.log_metric('accuracy',accuracy)
    # mlflow.log_param('max_depth', max_depth)
    # mlflow.log_param('n_estimators', n_estimators)

    # Creating Confusion_metrics
    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion_metrics')

    # Save plot
    plt.savefig("Confusion_matrics.png")

    # Log artifacts using mlflow
    # mlflow.log_artifact('Confusion_matrics.png')
    mlflow.log_artifact(__file__)

    # Set tags
    # mlflow.set_tags({"Author":"Shayan", "Project":"Wine_Classification"})

    # Log the model
    # mlflow.sklearn.log_model(rf, "Random forest Classification")

    print(accuracy)

