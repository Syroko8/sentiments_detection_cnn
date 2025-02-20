# Archivo que crea un modelo y lo guarda como un fichero.

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import pickle
import os

# 1. Cargamos los datos.
num_words = 10000   
max_len = 200      
embedding_dim = 128 

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words, test_size=0.2, random_state=42)

# 2. Preprocesamiento de los Datos.
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# 3. Construcción del Modelo CNN.

# 3.1 Definición de la red neuronal.
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_len))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))  

# 3.2 Compilación del modelo. 
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 4. Entrenamiento del modelo y evaluación.

print(">> Entrenando el modelo...")
model.fit(
    x_train, 
    y_train,
    epochs=3,       
    batch_size=128,
    validation_data=(x_train, y_test),
    verbose=1
)

print(">> Evaluación del modelo...")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f">> Exactitud del modelo: {acc*100:.2f}%")

# 5. Guardamos el modelo, que será usado por la aplicación cliente.

# Al no conseguir guardar el archivo en la carpeta de proyecto, obtendremos su dirección absoluta.
project_dir = os.getcwd() 
model_path = os.path.join(project_dir, "cnn_setiments_detection/sentiment_cnn.h5")
model.save(model_path)
print("Modelo guardado en 'sentiment_cnn.h5'.")

# 6. Guardamos el tokenizer.

word_index = imdb.get_word_index()

tokenizer_info = {
    'word_index': word_index,
    'num_words': num_words,
    'max_len': max_len
}

with open(os.path.join(project_dir, "cnn_setiments_detection/tokenizer_imdb.pkl"), "wb") as f:
    pickle.dump(tokenizer_info, f)

print("Tokenizer guardado en 'tokenizer_imdb.pkl'.")