import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
import random

class FlightPricePredictor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        self.model = LinearRegression()

    def load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path)

    def preprocess_data(self):
        self.dataset['data'] = pd.to_datetime(self.dataset['data'])
        self.dataset['dia_semana'] = self.dataset['data'].dt.dayofweek
        self.dataset['hora_partida'] = self.dataset['data'].dt.hour

    def split_data(self):
        X = self.dataset[['tempo_viagem', 'dia_semana', 'hora_partida']]
        y = self.dataset['preco_bilhete']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        return mse

    def predict_price(self, tempo_viagem, dia_semana, hora_partida):
        previsao = self.model.predict([[tempo_viagem, dia_semana, hora_partida]])
        return previsao[0]
