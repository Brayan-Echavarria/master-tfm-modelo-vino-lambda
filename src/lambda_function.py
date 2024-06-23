import json
import joblib
import numpy as np

model = joblib.load("/var/task/sklearn_model.joblib")

def lambda_handler(event, context):
    try:
        # Extraer los parámetros de entrada del evento
        params = json.loads(event['body'])
        fixed_acidity = float(params['fixed_acidity'])
        volatile_acidity = float(params['volatile_acidity'])
        citric_acid = float(params['citric_acid'])
        residual_sugar = float(params['residual_sugar'])
        chlorides = float(params['chlorides'])
        free_sulfur_dioxide = float(params['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(params['total_sulfur_dioxide'])
        density = float(params['density'])
        pH = float(params['pH'])
        sulphates = float(params['sulphates'])
        alcohol = float(params['alcohol'])

        # Crear el array de características
        features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                              chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                              pH, sulphates, alcohol]])

        # Realizar la predicción
        quality = model.predict(features)

        # Convertir el resultado a un tipo que pueda ser serializado por JSON
        quality = float(quality[0])

        return {
            'statusCode': 200,
            'body': json.dumps({'quality': quality})
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }


