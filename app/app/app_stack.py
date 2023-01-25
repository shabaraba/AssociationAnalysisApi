from aws_cdk import (
    Duration,
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as _apigateway,
)

from constructs import Construct


class AssociationAnalysisApiStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here
        my_lambda = _lambda.DockerImageFunction(
            self, 'AssociationAnalysisLambda',
            description='アソシエーション分析をするためのlambda',
            code= _lambda.DockerImageCode.from_image_asset('lambda'),
            timeout= Duration.seconds(60 * 5)
        )
        # create API Gateway
        api = _apigateway.LambdaRestApi(
            self, 'AssociationAnalysisApiGateway', handler=my_lambda, proxy=True)
        analysis_api = api.root.add_resource('analyze')
        analysis_api.add_method('POST')
