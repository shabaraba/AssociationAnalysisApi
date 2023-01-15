FROM amazon/aws-lambda-python:3.9
COPY ./ ./
RUN pip install -r requirements.txt
CMD ["association_analyze.handler"]