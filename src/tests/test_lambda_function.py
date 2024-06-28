import unittest
from unittest.mock import patch, Mock
import json
from lambda_function import lambda_handler

class TestLambdaHandler(unittest.TestCase):

    @patch('lambda_function.model.predict')
    def test_lambda_handler_success(self, mock_predict):
        mock_predict.return_value = [5.0]

        event = {
            'body': json.dumps([
                {
                    "fixed_acidity": 7.4,
                    "volatile_acidity": 0.7,
                    "citric_acid": 0.0,
                    "residual_sugar": 1.9,
                    "chlorides": 0.076,
                    "free_sulfur_dioxide": 11,
                    "total_sulfur_dioxide": 34,
                    "density": 0.9978,
                    "pH": 3.51,
                    "sulphates": 0.56,
                    "alcohol": 9.4
                }
            ])
        }

        response = lambda_handler(event, None)
        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        self.assertIn('qualities', body)
        self.assertEqual(body['qualities'], [5.0])

    @patch('lambda_function.model.predict')
    def test_lambda_handler_exception(self, mock_predict):
        mock_predict.side_effect = Exception("Test exception")

        event = {
            'body': json.dumps([
                {
                    "fixed_acidity": 7.4,
                    "volatile_acidity": 0.7,
                    "citric_acid": 0.0,
                    "residual_sugar": 1.9,
                    "chlorides": 0.076,
                    "free_sulfur_dioxide": 11,
                    "total_sulfur_dioxide": 34,
                    "density": 0.9978,
                    "pH": 3.51,
                    "sulphates": 0.56,
                    "alcohol": 9.4
                }
            ])
        }

        response = lambda_handler(event, None)
        self.assertEqual(response['statusCode'], 400)
        body = json.loads(response['body'])
        self.assertIn('error', body)

if __name__ == '__main__':
    unittest.main()
