# model_interface.py

class BaseModel:
    def train_model(self, data):
        raise NotImplementedError

    def predict(self, input_sequence):
        raise NotImplementedError
