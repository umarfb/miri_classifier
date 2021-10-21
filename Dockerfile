FROM python:3.8.12
COPY rnntools /opt/rnn_classifier/rnntools
COPY data /opt/rnn_classifier/data
COPY trained_models /opt/rnn_classifier/trained_models
COPY focal_loss.py mixed_input_train.py model_evaluation.py mixed_input_train_config.yaml requirements.txt /opt/rnn_classifier/
WORKDIR /opt/rnn_classifier
RUN pip install -r requirements.txt
ENTRYPOINT /bin/bash

