from flask import Flask, request, jsonify
import re
import load_bard
import load_llama

app = Flask(__name__)

# This is a simple Flask app that uses the Bart model to generate code completions
# It will start a port at 8080 and listen for POST requests at the /bart endpoint
# The input should be a JSON object with a key 'input_text' containing the code snippet
# The output will be a JSON object with a key 'prediction' containing the generated code completion

# You must download the model file from the following link and place it in the same directory as this script.
# Model link -> https://drive.google.com/file/d/1skgaVYnhBrB2ZnWgmOS0ZLVWe5QvH2Ar/view?usp=drive_link
# If you cannot access the model file, please contact David Wang for assistance (taweiw@andrew.cmu.edu).

# This class encapsulates the BART model and tokenizer

# Create an instance of the BartModel class
bart_model = load_bard.BartModel()
llama_model = load_llama.LlamaModel()

@app.route('/bart', methods=['POST'])
def predict_bart():
    data = request.json
    input_text = data['input_text']
    prediction = bart_model.get_prediction_test_by_input_string(input_text)
    return jsonify({'prediction': prediction})

@app.route('/llama', methods=['POST'])
def predict_llama():
    data = request.json
    fm = data['fm']
    fm_fc_ms_ff = data['fm_fc_ms_ff']
    prediction = llama_model.predict(fm, fm_fc_ms_ff)
    return jsonify({'prediction': prediction})

# Run this file to start the Flask app: python3 bart_flask_app.py
if __name__ == '__main__':
    port = 8080
    app.run(debug=True, port=port)
