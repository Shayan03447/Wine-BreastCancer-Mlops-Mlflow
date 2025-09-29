import mlflow
print("MLflow tracking uri is in the file format")
print(mlflow.get_tracking_uri())
print('\n')
mlflow.set_tracking_uri('http://127.0.0.1:5000')
print('Printing new tracking uri scheme below')
print(mlflow.get_tracking_uri())
print('\n')
