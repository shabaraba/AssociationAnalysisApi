import json

def handler(event, context):
    response = {'resp': 'hello!'}
    return {
        'statusCode': 200,
        'body': json.dumps(response),
    }