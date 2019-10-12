from keras.models import load_model
import numpy as np
import pandas as pd

data = pd.read_csv('data2.csv').to_numpy()
ac = load_model(r'.//weights//ae_weights.h5')
reconstructed_data = ac.predict(data)
print(reconstructed_data.shape)
