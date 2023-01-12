from aws_cdk import (
    # Duration,
    Stack,
    aws_lambda as _lambda,
    # aws_sqs as sqs,
)
from constructs import Construct

class AppStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here
        my_lambda = _lambda.Function(
            self, 'HelloHandler',
            runtime=_lambda.Runtime.PYTHON_3_7,
            code=_lambda.Code.from_asset('lambda'),
            handler='hello.handler',
        )

        # example resource
        # queue = sqs.Queue(
        #     self, "AppQueue",
        #     visibility_timeout=Duration.seconds(300),
        # )
