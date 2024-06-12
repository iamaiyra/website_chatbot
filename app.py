from flask import Flask, request, jsonify, render_template
import keras
import os
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from processes.dataset_creation import data_fetch_from_url
from processes.model_creation import create_model_from_clean_data


app = Flask(__name__)


def generate_response(seed_text, next_words):
    # Load the trained model
    model = keras.models.load_model(os.path.join('static', 'model', 'qa_model.h5'))

    # Tokenizer (should be the same as used during training)
    tokenizer = Tokenizer()
    with open(os.path.join('static', 'data', 'cleaned_data.txt'), 'r', encoding='utf-8') as file:
        data = file.readlines()
    tokenizer.fit_on_texts(data)

    output_word = ""
    for _ in range(next_words):
        input_sequences = []
        for line in seed_text.split('.'):
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        # Pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])

        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        # predicted = model.predict(token_list, verbose=0)
        print("predicted", tokenizer.word_index)

        for word, index in tokenizer.word_index.items():
            output_word += " " + word
        break
    seed_text += " " + output_word
    return seed_text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    response = generate_response(user_input, next_words=20)
    return jsonify({'response': response})


if __name__ == '__main__':
    print("Fetching and cleaning data")
    # data_fetch_from_url()
    print("loading model")
    # create_model_from_clean_data()
    app.run(host="0.0.0.0", port=5006, debug=True, use_reloader=False)
