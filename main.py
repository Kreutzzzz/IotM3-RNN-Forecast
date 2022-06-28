# %% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 20)
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')


## Leitura da base da dados
data = pd.read_csv("city_temperature.csv")
data.head()

## Definindo a cidade de são paulo para a analise
sao_paulo = data[data["City"] == "Sao Paulo"]
sao_paulo.head()

## Prenchendo os valores que faltam
sao_paulo["AvgTemperature"] = np.where(sao_paulo["AvgTemperature"] == -99, np.nan, sao_paulo["AvgTemperature"])
sao_paulo.isnull().sum()

sao_paulo["AvgTemperature"] = sao_paulo["AvgTemperature"].ffill()
sao_paulo.isnull().sum()

## Criando uma coluna de tempo para melhor visualização no grafico
sao_paulo.dtypes
sao_paulo["Time_steps"] = pd.to_datetime((sao_paulo.Year*10000 + sao_paulo.Month*100 + sao_paulo.Day).apply(str),format='%Y%m%d')
sao_paulo['AvgTemperature']=sao_paulo['AvgTemperature'].apply(lambda x : ((x - 32) * (5/9)))
sao_paulo.head()

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Ano")
    plt.ylabel("Temperatura")
    plt.grid(True)

## Cirando as series
time_step = sao_paulo["Time_steps"].tolist()
temperatura = sao_paulo["AvgTemperature"].tolist()

series = np.array(temperatura)
time = np.array(time_step)

#%% Visualização da série de validação
plt.figure(figsize=(10, 6))
print(len(series))
plot_series(time[8000:], series[8000:])

# %% 
plt.figure(figsize=(10, 6))
plot_series(time[-365:], series[-365:])

## Definindo o a serie de treinamento e a de validação
## sendo a base com 9266 valores com os primeiros 8000 para treinamento e o restante para validação
split_time = 8000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


# %% Criando os tensores
series1 = tf.expand_dims(series, axis=-1)
ds = tf.data.Dataset.from_tensor_slices(series1[:20])
for val in ds:
    print(val.numpy())

dataset = ds.window(5, shift=1)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()

## Agrupando os valores em grupos de 5 em uma unica coluna para melhor otimização
dataset = ds.window(5, shift=1, drop_remainder=True)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()

dataset = ds.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
    print(window.numpy())

dataset = ds.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x,y in dataset:
    print(x.numpy(), y.numpy())

dataset = ds.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
for x,y in dataset:
    print(x.numpy(), y.numpy())

dataset = ds.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())
    print("*"*25)


# %% 
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)



tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])


optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set,epochs=500)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
    
window_size = 64
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

# %% Comparação da previsão com a validação
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)

# %% Comparação analisando somente o ultimo ano

plt.figure(figsize=(10, 6))
plot_series(time_valid[-365:], x_valid[-365:])
plot_series(time_valid[-365:], rnn_forecast[-365:])

# %% Visualização do erro

tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
loss=history.history['loss']

epochs=range(len(loss))

plt.plot(epochs, loss, 'r')
plt.title('Erro')
plt.xlabel("Ciclo")
plt.ylabel("Erro")
plt.legend(["Erro"])
plt.figure()

zoomed_loss = loss[200:]
zoomed_epochs = range(200,500)

plt.plot(zoomed_epochs, zoomed_loss, 'r')
plt.title('Erro')
plt.xlabel("Ciclo")
plt.ylabel("Erro")
plt.legend(["Erro"])
plt.figure()

# %%
