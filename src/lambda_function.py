import json

def lambda_handler(event, context):
    response = {
        "statusCode": 200,
        "body": json.dumps({"message": "Hello from Lambda!"})
    }
    return response
