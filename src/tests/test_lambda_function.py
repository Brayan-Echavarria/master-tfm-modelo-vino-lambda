import unittest
from unittest.mock import patch, Mock
import json
import os

# Asegúrate de que lambda_function está en el path correcto
import lambda_function

class TestLambdaHandler(unittest.TestCase):

    def setUp(self):
        # Configurar la variable de entorno para la ruta del modelo
        os.environ['MODEL_PATH'] = './sklearn_model.joblib'

    @patch('lambda_function.joblib.load')
    def test_lambda_handler_success(self, mock_load):
        # Configurar el mock del modelo
        mock_model = Mock()
        mock_model.predict.return_value = [5.0]
        mock_load.return_value = mock_model

        # Crear un evento de prueba
        event = {
            'body': json.dumps([
                {
                    "fixed_acidity": 7.9,
                    "volatile_acidity": 0.43,
                    "citric_acid": 0.21,
                    "residual_sugar": 1.6,
                    "chlorides": 0.106,
                    "free_sulfur_dioxide": 10,
                    "total_sulfur_dioxide": 37,
                    "density": 0.9966,
                    "pH": 3.17,
                    "sulphates": 0.91,
                    "alcohol": 9.5
                }
            ])
        }

        # Llamar a la función lambda_handler
        response = lambda_function.lambda_handler(event, None)

        # Verificar la respuesta
        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        self.assertIn('qualities', body)
        self.assertEqual(body['qualities'], [5.0])

    @patch('lambda_function.joblib.load')
    def test_lambda_handler_invalid_input(self, mock_load):
        # Configurar el mock del modelo
        mock_load.return_value = Mock()

        # Crear un evento de prueba con entrada inválida
        event = {
            'body': json.dumps({"fixed_acidity": 7.9})
        }

        # Llamar a la función lambda_handler
        response = lambda_function.lambda_handler(event, None)

        # Verificar la respuesta
        self.assertEqual(response['statusCode'], 400)
        body = json.loads(response['body'])
        self.assertIn('error', body)

    @patch('lambda_function.joblib.load', side_effect=Exception('Test exception'))
    def test_lambda_handler_exception(self, mock_load):
        # Crear un evento de prueba
        event = {
            'body': json.dumps([
                {
                    "fixed_acidity": 7.9,
                    "volatile_acidity": 0.43,
                    "citric_acid": 0.21,
                    "residual_sugar": 1.6,
                    "chlorides": 0.106,
                    "free_sulfur_dioxide": 10,
                    "total_sulfur_dioxide": 37,
                    "density": 0.9966,
                    "pH": 3.17,
                    "sulphates": 0.91,
                    "alcohol": 9.5
                }
            ])
        }

        # Llamar a la función lambda_handler
        response = lambda_function.lambda_handler(event, None)

        # Verificar la respuesta
        self.assertEqual(response['statusCode'], 400)
        body = json.loads(response['body'])
        self.assertIn('error', body)
        self.assertEqual(body['error'], 'Test exception')

if __name__ == '__main__':
    unittest.main()
