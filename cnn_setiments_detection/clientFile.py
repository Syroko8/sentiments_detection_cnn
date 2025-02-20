from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
import pickle
import os

app = Flask(__name__)

# 1. Cargar el modelo

# Obtenemos la ruta de ejecución del fichero.
project_dir = os.getcwd() 
model_path = os.path.join(project_dir, "cnn_setiments_detection/sentiment_cnn.h5")
model = tf.keras.models.load_model(model_path)

# 2. Cargar tokenizer info (sin la función text_to_sequence)
project_dir = os.getcwd() 
tokenizer_path = os.path.join(project_dir, "cnn_setiments_detection/tokenizer_imdb.pkl")
with open(tokenizer_path, "rb") as f:
    tokenizer_info = pickle.load(f)

word_index = tokenizer_info['word_index']
num_words = tokenizer_info['num_words']
max_len = tokenizer_info['max_len']

# Definir text_to_sequence en app.py
def text_to_sequence(text):
    tokens = text.lower().split()
    seq = []
    for token in tokens:
        if token in word_index and word_index[token] < num_words:
            seq.append(word_index[token] + 3)  
        else:
            seq.append(2) 
    return seq

# 3. Plantilla HTML.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Análisis de Sentimientos</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: auto; }
        textarea { width: 100%; height: 100px; }
        .result { font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Análisis de Sentimientos (CNN - IMDB)</h1>
        <form method="post" action="/">
            <label for="review">Ingresa tu reseña de película:</label><br><br>
            <textarea name="review" id="review" required></textarea><br><br>
            <button type="submit">Analizar</button>
        </form>
        
        {% if prediction is not none %}
            <div class="result">
                <p>Resultado: {{ prediction }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    
    if request.method == "POST":
        user_review = request.form.get("review", "")
        
        # Preprocesar texto
        seq = text_to_sequence(user_review)

        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = [0]*(max_len - len(seq)) + seq  # rellenamos al principio
        
        seq_array = np.array([seq])  # shape (1, max_len)

        # Hacer predicción
        prob = model.predict(seq_array)[0][0] 

        if prob >= 0.5:
            prediction = f"Positiva (confianza: {prob:.2f})"
        else:
            prediction = f"Negativa (confianza: {prob:.2f})"
    
    # Renderizar la plantilla con Flask
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    # Ejecutar la app en modo debug
    app.run(debug=True)
