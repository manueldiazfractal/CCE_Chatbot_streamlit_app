import streamlit as st
import replicate
import os
import pandas as pd
import numpy as np
from PIL import Image

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Cargar el modelo de spaCy
nlp = spacy.load('es_core_news_sm')

df = pd.read_csv("./data/data for streamlit.csv")
df = df.dropna().reset_index(drop = True)

# Función para extraer la intención
def extract_intent(text):
    doc = nlp(text)
    intent = [token.text for token in doc if token.pos_ == 'NOUN']
    return '-'.join(intent)

def truncate_to_token_limit(text, token_limit=2000):
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:token_limit]
    return tokenizer.convert_tokens_to_string(truncated_tokens)


def search(query, top_k=2):
    # Extraer la intención de la consulta del usuario
    user_intent = extract_intent(query)

    # Calcular la similitud coseno entre las intenciones usando TF-IDF
    vectorizer = TfidfVectorizer()
    intents_matrix = vectorizer.fit_transform(df['intent'].tolist() + [user_intent])
    cosine_similarities = linear_kernel(intents_matrix[-1], intents_matrix[:-1]).flatten()

    # Obtener los mejores resultados según la similitud coseno
    top_results = cosine_similarities.argsort()[:-top_k-1:-1]

    results_text = f"\nTop {top_k} results, based on intent matching:\n"

    meta_titles = []  # Lista para almacenar los títulos de los documentos

    for idx in top_results:
        meta_title = df.iloc[idx]['ConcatenatedString']
        meta_titles.append(meta_title)  # Añade el título a la lista
        description = df.iloc[idx]['combined_text']
        truncated_description = truncate_to_token_limit(description, 2400)
        results_text += truncated_description + "\n\n"

    return results_text, meta_titles  # Devuelve los títulos de los documentos junto con el texto de los resultados

# Configurar la llamada a Replicate con LLaMA
os.environ['REPLICATE_API_TOKEN'] = 'r8_ZoboQCK8ShD2AE31PSqZtmtvzTjujbw0qaD1T'
api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

# api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")

# question = '¿Cómo puedo cuidar la pintura de mi coche?'



# App title
st.set_page_config(page_title="🦙💬 Llama2/BERT AWS Sagemaker Assistant Assistant")

# Replicate Credentials
with st.sidebar:
    
    image = Image.open('carcareeurope.jpg')
        
    # Muestra la imagen en el sidebar con un ancho de 100 píxeles
    st.image(image, width=300)
    
    
    st.title('🦙💬 Llama2 Car Care Europe ChatBot')
    

    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    

    # replicate_api = st.text_input('Ingresa Replicate API token:', type='password')
    # if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
    #     st.warning('Por favor Ingresa las Credenciales!', icon='⚠️')
    # else:
    #     st.success('Ahora puedes ingresar el promt del mensaje!', icon='👉')
        
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Modelos y Parametros')
    selected_model = st.sidebar.selectbox('Elige Modelo de Llama2', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "¿En qué puedo ayudarle?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Limpiar Historial de Chat', on_click=clear_chat_history)



system_prompt = """You are an assistant focused on helping Car Care Europe's customers about automotive cleaning, 
maintenance and enhancement products. You can answer about the context, but also add your own knowledge, 
and you must answer in Spanish.
"""

def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    # Agregar la búsqueda de contexto a la entrada del prompt
    search_result, filenames = search(prompt_input)  # Obtiene los nombres de los archivos de la función search
    prompt_input = "Question: " + prompt_input + "\n\n" + search_result

    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "debug": False,
            "top_k": 50,
            "top_p": top_p,
            "prompt": prompt_input,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "max_new_tokens": 500,
            "min_new_tokens": -1
            }
        )

    return output, filenames


# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, filenames = generate_llama2_response(prompt)  # Obtiene los nombres de los archivos de la función generate_llama2_response
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            # Añade los nombres de los archivos como "fuentes" al final de la respuesta
            full_response += "\n" + "\nLinks de Interes:\n" + "\n".join("* " + filename for filename in filenames)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

