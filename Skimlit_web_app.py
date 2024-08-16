import tensorflow as tf
import pickle
import streamlit as st

def load_model_and_vocab(model_path, char_vocab_path, token_vocab_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load the vocabularies
    with open(char_vocab_path, 'rb') as f:
        char_vocab = pickle.load(f)

    with open(token_vocab_path, 'rb') as f:
        token_vocab = pickle.load(f)

    # Get the vectorizers from the model
    char_vectorizer = model.get_layer("vectorizer_layer")
    token_vectorizer = model.get_layer("token_vectorization_layer")

    # Re-adapt the vectorizers with the saved vocabulary
    char_vectorizer.set_vocabulary(char_vocab)
    token_vectorizer.set_vocabulary(token_vocab)

    return model

# Function to convert prediction to class label
def convert_prediction_to_class(prediction, class_labels):
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    return class_labels[predicted_class_index]


st.title("Skimlit Application")

user_input = st.text_area("Enter a paragraph:",height=200)

if st.button("Predict") and user_input:
    # Load model and vocab
    model_path = "Skimlit_project_model.h5"
    char_vocab_path = "char_vocab.pkl"
    token_vocab_path = "token_vocab.pkl"
    
    model = load_model_and_vocab(model_path, char_vocab_path, token_vocab_path)
    
    # Split the input into sentences
    sentences = user_input.split(". ")  
    
    # Initialize dictionaries to store sentences by predicted class
    classified_sentences = {label: [] for label in ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']}
    
    for sentence in sentences:
        if sentence.strip():  # Check if sentence is not empty
            
            input_1 = [sentence]  # Token-based processing input
            input_2 = [" ".join(list(sentence))]  # Char-based processing input
            
            # Converting inputs to tensors
            input_1_tensor = tf.convert_to_tensor(input_1, dtype=tf.string)
            input_2_tensor = tf.convert_to_tensor(input_2, dtype=tf.string)
            
            # Expand dimensions to match model input shape
            input_1_tensor = tf.expand_dims(input_1_tensor, axis=0)  
            input_2_tensor = tf.expand_dims(input_2_tensor, axis=0) 
            
            predictions = model.predict({"char_inputs": input_2_tensor, "token_inputs": input_1_tensor})
            
            # Class labels
            class_labels = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
            
            # Convert predictions to class labels
            predicted_class = convert_prediction_to_class(predictions, class_labels)
            
            # Append sentence to the category
            classified_sentences[predicted_class].append(sentence)
    
    st.write("**Results:**")
    structured_output = ""
    for label in ['OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'BACKGROUND']: 
        if classified_sentences[label]:
            structured_output += f"**{label.capitalize()}:** {'. '.join(classified_sentences[label])}\n\n"
    
    st.write(structured_output)
