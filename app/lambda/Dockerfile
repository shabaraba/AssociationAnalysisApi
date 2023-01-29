FROM amazon/aws-lambda-python:3.9
COPY ./ ./
RUN /var/lang/bin/python3.9 -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["association_analyze.handler"]