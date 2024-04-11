# model_interface.py

class BaseModel:
    def train_model(self, train_data, test_data):
        raise NotImplementedError

    def predict(self, input_sequence):
        raise NotImplementedError
