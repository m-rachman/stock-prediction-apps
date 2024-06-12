from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
import tensorflow as tf

def model_train (x_train, y_train):
  tf.random.set_seed(100)
  tf.keras.backend.clear_session()

  model = Sequential()
  model.add(Bidirectional(GRU(units=256, activation='relu', return_sequences=True, input_shape=(10,1))))
  model.add(Dropout(0.5))
  model.add(Bidirectional(GRU(units=256, activation='relu', return_sequences=False, kernel_regularizer=L2(0.01))))
  model.add(Dropout(0.1))
  model.add(Dense(units=1, activation='linear')) # Prediction of the next value

  model.compile(optimizer='adam', loss='mape', metrics=['mape'])

  model.build((None, 10, 1))  # Specify input shape

  # Define EarlyStopping callback
  early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

  # Train the model
  history = model.fit(x_train, y_train, epochs=10, batch_size=20, validation_split=0.2, callbacks=[early_stopping])
  return history, model

def model_predict(train,scaler,model):
  data_inf = train[-10:]
  scaled_data_inf = scaler.transform(data_inf.values.reshape(-1,1))
  data_inf_final = tf.expand_dims(scaled_data_inf,0)
  pred_inf = model.predict(data_inf_final)
  return scaler.inverse_transform(pred_inf) , data_inf

