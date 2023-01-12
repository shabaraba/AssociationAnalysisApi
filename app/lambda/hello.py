import json

def handler(event, context):
    print('request: {}'.format(json.dumps(event)))
    return {
        'status': 200,
        'headers': {
            'Content-Type': 'text/plain'
        },
        'body': 'hello, cdk! you have hit {} \n'.format(event['path'])
    }
    