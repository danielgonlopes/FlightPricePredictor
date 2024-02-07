import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Gerar dados fictícios
num_samples = 1000
np.random.seed(42)

data = {
    'tempo_percurso': np.random.randint(5, 60, num_samples),  # Tempo de percurso em minutos
    'preco_final': np.random.uniform(5, 50, num_samples),  # Preço final da corrida
    'data': [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365),
                                               hours=random.randint(0, 23),
                                               minutes=random.randint(0, 59))
             for _ in range(num_samples)]  # Data e hora de início da corrida
}

# Criar DataFrame
df = pd.DataFrame(data)

# Salvar o DataFrame em um arquivo CSV
df.to_csv('data/dataset_flight.csv', index=False)
