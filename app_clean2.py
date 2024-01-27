import streamlit as st
import replicate
import os
import pandas as pd
import numpy as np
from PIL import Image
import transformers
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords

# os.chdir(r"C:\Users\MANUEL ALEJANDRO\Documentos\Fractal")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Cargar el modelo de spaCy para procesamiento de lenguaje natural en español
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
nlp = spacy.load('es_core_news_sm')

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Carga el tokenizador de Roberta para procesamiento de texto
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Cargar datos desde un archivo CSV
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
df = pd.read_excel("./data/data for streamlit2.xlsx")
# df = df.dropna().reset_index(drop = True)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Función para extraer la intención de un texto usando el modelo de spaCy
# Utiliza el modelo de spaCy para identificar sustantivos en el texto,
# los cuales se consideran como la "intención" detrás de la consulta del usuario.
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def extract_intent(text):
    doc = nlp(text)
    intent = [token.text for token in doc if token.pos_ == 'NOUN']
    # Une los sustantivos con guiones para formar una cadena de intención
    return '-'.join(intent)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Función para truncar un texto a un límite de tokens.
# Utiliza el tokenizador de Roberta para dividir el texto en tokens y luego truncar.
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def truncate_to_token_limit(text, token_limit=2000):
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:token_limit]
    # Convierte los tokens truncados de nuevo a una cadena de texto
    return tokenizer.convert_tokens_to_string(truncated_tokens)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Función para filtrar un DataFrame por palabras clave
# Utiliza las palabras clave de la consulta del usuario para filtrar productos relevantes.
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def filter_by_keywords(df, query, keyword_columns):
    """
    Filtra el dataframe por palabras clave presentes en las columnas especificadas,
    eliminando las stopwords.
    """
    # Define las stopwords en español
    stop_words = set(stopwords.words('spanish'))

    # Divide la consulta en palabras y elimina las stopwords
    keywords = [word for word in query.lower().split() if word not in stop_words]
    #  Aplica un filtro al DataFrame para seleccionar filas que contienen las palabras clave
    mask = df[keyword_columns].apply(lambda x: any(keyword in str(x).lower() for keyword in keywords), axis=1)
    return df[mask].drop_duplicates(subset=["product_name", "categories", "description"])

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Función para buscar productos relevantes en base a una consulta
# Calcula la similitud de las intenciones entre la consulta del usuario y las descripciones de los productos.
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def search(query, top_k=3):
    # Filtrar primero por palabras clave
    keyword_columns = ["product_name", "categories", "description"]
    filtered_df = filter_by_keywords(df, query, keyword_columns)
    
    # Si no se encuentran resultados, retorna un mensaje indicándolo
    if filtered_df.empty:
        return "No se encontraron resultados con las palabras clave especificadas.", []

    # Extraer la intención de la consulta del usuario
    user_intent = extract_intent(query)

    # Calcular la similitud coseno entre las intenciones usando TF-IDF
    vectorizer = TfidfVectorizer()
    intents_matrix = vectorizer.fit_transform(filtered_df['combined_text'].tolist() + [user_intent])
    cosine_similarities = linear_kernel(intents_matrix[-1], intents_matrix[:-1]).flatten()

    # Obtener los mejores resultados según la similitud coseno
    filtered_df['similarity'] = cosine_similarities
    sorted_df = filtered_df.sort_values(by=['similarity', 'ordered_qty', 'price'], ascending=[False, False, True])

    top_results = sorted_df.head(top_k)

    results_text = f"\n{top_k} products results:\n"
    meta_titles = []  # Lista para almacenar los títulos de los documentos

    # Contador para enumerar los resultados
    # counter = 1
    # Itera sobre los resultados y los agrega al texto de salida
    for _, row in top_results.iterrows():
        meta_title = row['ConcatenatedString']
        meta_titles.append(meta_title)  # Añade el título a la lista
        description = row['combined_text']
        truncated_description = truncate_to_token_limit(description, 1500)
        results_text += f"{truncated_description}\n"   # Añade la numeración: f"{counter}. {truncated_description}\n\n"
        # counter += 1  # Incrementa el contador

     # Retorna el texto de los resultados y los títulos para su uso posterior
    return results_text, meta_titles

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Prompt para clasificar las entradas de los usuarios en preguntas o despedidas
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# # promt con tres opciones, No funciona bien
# system_prompt_classify = """
# As an AI, classify Spanish user inputs into three categories: 'question' for car product inquiries, 'farewell' for goodbyes, or 'no understand' if the input is unclear or difficult to comprehend. 
# Analyze each input and respond with only one of these specific words. Refrain from providing additional information or elaboration.
# """

system_prompt_classify = """
Your task as an AI is to classify Spanish user inputs into two categories: 'question' for inquiries about car products, or 'farewell' for goodbyes.
Analyze the input and respond with only one of these words. Avoid extra information or elaboration.
"""

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Configurar la autenticación para la API de Replicate
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
os.environ['REPLICATE_API_TOKEN'] = 'r8_2XoQonAq3JvwRrjFOFD84EEfUJaXxCr3QLEUw'
# r8_ZoboQCK8ShD2AE31PSqZtmtvzTjujbw0qaD1T
# r8_2XoQonAq3JvwRrjFOFD84EEfUJaXxCr3QLEUw
api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Función para clasificar la entrada del usuario usando el modelo Llama 2.
# Determina si la entrada es una pregunta, una despedida, o si no se entiende.
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def classify_input(question, api):
    promt = "User Input: " + question
    #  Ejecuta el modelo Llama 2 con el prompt y el system_prompt_classify
    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "debug": False,
            "top_k": 50,
            "top_p": 1,
            "prompt": promt,
            "temperature": 0.5,
            "system_prompt": system_prompt_classify,
            "max_new_tokens": 500,
            "min_new_tokens": -1
        }
    )

    # Concatena la salida del modelo y la devuelve
    out = []
    for value in output:
        out.append(value)
    
    text_answer = ''.join(out).strip()

    # # Clasifica la respuesta como 'pregunta' o 'despedida'
    # if "pregunta" in text_answer:
    #     return "pregunta"
    # elif "despedida" in text_answer:
    #     return "despedida"
    # else:
    #     return "indefinido"  # En caso de que la respuesta no sea clara
    return text_answer


# classify_input("Conoces el producto Gummi", api).lower()

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Configuración de la página de Streamlit
# App title
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
st.set_page_config(page_title="🦙💬 Llama2/BERT AWS Sagemaker Assistant Assistant")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Configuración de la barra lateral en Streamlit
# Replicate Credentials
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
with st.sidebar:
    
    image = Image.open('carcareeurope.jpg')
        
    # Muestra la imagen en el sidebar con un ancho de 100 píxeles
    st.image(image, width=300)
    
    
    st.title('🦙💬 Llama2 Car Care Europe ChatBot')
    

    # if 'REPLICATE_API_TOKEN' in st.secrets:
    #     st.success('API key already provided!', icon='✅')
    #     replicate_api = st.secrets['REPLICATE_API_TOKEN']
    # else:
    #     replicate_api = st.text_input('Enter Replicate API token:', type='password')
    #     if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
    #         st.warning('Please enter your credentials!', icon='⚠️')
    #     else:
    #         st.success('Proceed to entering your prompt message!', icon='👉')
    

    replicate_api = st.text_input('Ingresa Replicate API token:', type='password')
    if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
        st.warning('Por favor Ingresa las Credenciales!', icon='⚠️')
    else:
        st.success('Ahora puedes ingresar el promt del mensaje!', icon='👉')
        
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

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Inicialización del historial de mensajes
# Store LLM generated responses
#
# NOTA: No funciona bien, se debe porbar otra forma de almacenar la 
#       memoria del chatbot, ej: LlamaIndex o LangChain
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hola, me alegra que estés en nuestra tienda 🤗. ¿En qué puedo ayudarle?"}]

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Mostrar o limpiar mensajes de chat
# Display or clear chat messages
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Limpiar Historial de Chat', on_click=clear_chat_history)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Prompt del sistema para generación de respuestas
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
system_prompt = """You must always respond in Spanish and Don't say hello when you start answering. As an assistant, you will help customers to buy the best articles for the car. 
Keep in mind that the user does not know anything about the products or the context.
You will only answer based on the context of the description of the products, taking into account the Question. 
Guide users by comparing items and highlighting popular ones, recommending suitability, and providing prices in Euros. 
If you are unclear about the query, ask for clarification. 
Conclude by checking if further help is needed, and bid farewell politely if the conversation ends.
"""


questions = ["question", "query", "inquiry", ]
farewells = ["farewell", "goodbye", "see you"]

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Función para generar respuestas de Llama2
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def generate_llama2_response(prompt_input):
    classification = classify_input(prompt_input, api)  # Usa la función classify_input
    classification = classification.lower()

    if classification in questions:
        # Si es una pregunta, realiza la búsqueda y prepara el prompt para la respuesta
        search_result, filenames = search(prompt_input)
        prompt_input = "Question: " + prompt_input + "\n\n" + search_result
    elif classification in farewells:
        # Si es una despedida, prepara un mensaje de despedida
        return ["Gracias por contactarnos. ¡Hasta pronto!"], []
    else:
        # Maneja casos donde la clasificación no es clara
        return ["Lo siento, no entendí tu pregunta o comentario."], []

    # Si es una pregunta, procede con la generación de la respuesta
    output = replicate.run('meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3', 
        input={
            "debug": False,
            "top_k": 50,
            "top_p": top_p,
            "prompt": prompt_input,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "max_new_tokens": 500,
            "min_new_tokens": -1
            })

    return output, filenames

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Captura y procesamiento de entradas del usuario
# User-provided prompt
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Generación de una nueva respuesta si el último mensaje no es del asistente
# Generate a new response if last message is not from assistant
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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