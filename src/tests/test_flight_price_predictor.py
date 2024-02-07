import unittest
from flight_price_predictor import FlightPricePredictor

class TestFlightPricePredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = FlightPricePredictor('../data/dataset_flight.csv')
        self.predictor.load_dataset()
        self.predictor.preprocess_data()
        self.predictor.split_data()
        self.predictor.train_model()

    def test_model_evaluation(self):
        mse = self.predictor.evaluate_model()
        self.assertIsInstance(mse, float)

    def test_predict_price(self):
        dia_semana = 0  # Segunda-feira
        hora_partida = 8  # 8:00 da manhã
        tempo_viagem = 120  # Tempo de viagem em minutos
        previsao = self.predictor.predict_price(tempo_viagem, dia_semana, hora_partida)
        self.assertIsInstance(previsao, float)

    def tearDown(self):
        del self.predictor

if __name__ == '__main__':
    unittest.main()
