# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import CSVLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from flask import Flask, render_template, request, jsonify,send_file
# from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelWithLMHead, pipeline, CLIPTokenizerFast, CLIPProcessor, CLIPModel,T5Tokenizer,MT5ForConditionalGeneration,Text2TextGenerationPipeline
# from langchain.prompts import PromptTemplate
# from PIL import Image
# import torch
# import numpy as np
# import csv
# import re
# import base64
# import speech_recognition as sr
# from gtts import gTTS
# import io
# from lingua import Language, LanguageDetectorBuilder
# import whisper
# from waitress import serve
# from pydub import AudioSegment
# import io
# from flask_cors import CORS

# load_dotenv()
# STATIC_URL_PATH = "/static"
# app = Flask(__name__,static_url_path=STATIC_URL_PATH)
# CORS(app)

# languages = [Language.ENGLISH, Language.MALAY, Language.CHINESE]
# detector = LanguageDetectorBuilder.from_languages(*languages).build()

# prompt_template = """
# As a responsible,super loyal and multilingual(ENGLISH,CHINESE,MALAY) e-library assistant highly focused on u-Pustaka, the best e-library in Malaysia, I will use the following information to answer your question:
# **Context:**
# {context}
# **Question:**
# {question}
# **Answer Guidelines:**
# * I will base my answer solely on the facts provided in the document. 
# * If the document doesn't contain the answer, I will honestly say "I don't know" instead of making something up.
# * My response will adhere to u-Pustaka's high standards and avoid:
#     * Pornography or violence
#     * Negative or false information
#     * Mentioning weaknesses or suggesting improvements for u-Pustaka
#     * Referencing or comparing u-Pustaka to competitors
#     * Referencing other irrelevent or private sources

# **Answer:**

# """

# prompt = PromptTemplate(template=prompt_template,
#                         input_variables=["context", "question"])

# MODEL_ID = "openai/clip-vit-base-patch32"
# en_to_zh_model = "K024/mt5-zh-ja-en-trimmed"
# zh_to_en_model = 'liam168/trans-opus-mt-zh-en'

# model_list = None
# vectorstore = None
# retriever = None
# qa = None
# device = None
# Clp_model = None
# tokenizer1 = None
# language_detect_model = None
# pipe = None
# image_arr = {}
# r = sr.Recognizer()
# custom_glossary = {}
# custom_glossary_bm = {}
# q_a = {}
# qa_index = [-1,-1,-1]
# PATTERN = re.compile(r'u?-?pustaka', re.IGNORECASE)


# @app.route('/', methods=['GET'])
# def home_page():
#     return render_template("index.html")

# def intialize():
#     global model_list, vectorstore, retriever, qa, device, Clp_model, tokenizer1, image_arr,pipe,language_detect_model

#     ms_tokenizer = AutoTokenizer.from_pretrained(
#         'mesolitica/translation-t5-small-standard-bahasa-cased-v2',
#         use_fast=False)
#     ms_model = T5ForConditionalGeneration.from_pretrained(
#         'mesolitica/translation-t5-small-standard-bahasa-cased-v2')
#     zh_model1 = AutoModelWithLMHead.from_pretrained(zh_to_en_model)
#     zh_tokenizer1 = AutoTokenizer.from_pretrained(zh_to_en_model)
#     embeddings_model_name = "BAAI/bge-small-en"
#     model_kwargs = {"device": "cpu"}
#     encode_kwargs = {"normalize_embeddings": True}
#     embeddings_model = HuggingFaceBgeEmbeddings(
#         model_name=embeddings_model_name,
#         model_kwargs=model_kwargs,
#         encode_kwargs=encode_kwargs)

#     model_list = [
#         embeddings_model, ms_tokenizer, ms_model,zh_tokenizer1, zh_model1
#     ]

#     loader = CSVLoader(file_path="./csvFile/en-FAQ.csv")
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000,
#                                           chunk_overlap=30,
#                                           separator="\n")
#     docs = text_splitter.split_documents(documents=documents)

#     vectorstore = FAISS.from_documents(docs, embeddings_model)
#     vectorstore.save_local("faiss_index_react")

#     new_vectorstore = FAISS.load_local("faiss_index_react",
#                                        embeddings_model,
#                                        allow_dangerous_deserialization=True)
#     retriever = new_vectorstore.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={
#             "score_threshold": 0.6,
#             "k": 5
#         })
#     qa = RetrievalQA.from_chain_type(llm=GoogleGenerativeAI(
#         model="models/text-bison-001", temperature=0),
#                                      chain_type_kwargs={"prompt": prompt},
#                                      retriever=retriever)

#     device = "cuda" if torch.cuda.is_available() else (
#         "mps" if torch.backends.mps.is_available() else "cpu")
    
#     language_detect_model = whisper.load_model("base")
#     Clp_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
#     tokenizer1 = CLIPTokenizerFast.from_pretrained(MODEL_ID)
#     image_arr = embed_image()
    
#     pipe = Text2TextGenerationPipeline(
#         model=MT5ForConditionalGeneration.from_pretrained(en_to_zh_model),
#         tokenizer=T5Tokenizer.from_pretrained(en_to_zh_model),
#     )
    
#     with open('./csvFile/glossary.csv', mode='r', encoding='utf-8') as csv_file:
#         reader = csv.DictReader(csv_file)
#         for row in reader:
#             custom_glossary[row['source']] = row['target']
    
#     with open("./csvFile/en-FAQ.csv",encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             q_a[row['Question']] = row['Answer']
            
#     with open("./csvFile/glossary_bm.csv",encoding='utf-8') as file2:
#         reader = csv.DictReader(file2)
#         for row in reader:
#             custom_glossary_bm[row['source']] = row['target']
            
#     csv_file.close()
#     file.close()
#     file2.close()
        

# def embed_image():

#     # List all files in the folder
#     files = os.listdir("./image")
#     img_emb = {}

#     # Loop through each file
#     for file_name in files:
#         # Check if the file is an image file
#         if file_name.endswith(('jpeg', 'png', 'jpg', 'gif')):
#             # Open the image using PIL
#             image_path = os.path.join("./image", file_name)
#             processor = CLIPProcessor.from_pretrained(MODEL_ID)
#             image_exp = Image.open(image_path)

#             image = processor(text=None, images=image_exp,
#                             return_tensors='pt')['pixel_values'].to(device)

#             image_emb = Clp_model.get_image_features(image)
#             image_emb = image_emb.cpu().detach().numpy()
#             image_arr = image_emb.T / np.linalg.norm(image_emb, axis=1)
#             image_arr = image_arr.T
#             img_emb[image_path] = image_arr

#             image_exp.close()

#     return img_emb


# def embed_text(user_msg):
#     global tokenizer1

#     inputs = tokenizer1(user_msg, return_tensors="pt")
#     text_emb = Clp_model.get_text_features(**inputs)
#     text_emb = text_emb.cpu().detach().numpy()
#     return text_emb


# def should_retrieve_map(user_msg):
#     text_emb = embed_text(user_msg)
#     max = 0
#     for image_path,value in image_arr.items():
#         scores = np.dot(text_emb, value.T)[0][0]
#         if scores > max:
#             max = scores
#             fp = image_path
#     if max > 3.1:
#         with open(fp, "rb") as img_file:
#             image_data = base64.b64encode(img_file.read()).decode('utf-8')
#         return image_data
#     return None


# @app.route('/process_user_message', methods=['GET','POST'])
# def process_user_message():
#     global model_list, qa
#     bot_response = ''
#     data = request.get_json()
#     user_msg = data.get('user_msg')
#     index = data.get('number')
    
#     filtered_msg = re.sub(PATTERN, '', user_msg)
#     language = detector.detect_language_of(filtered_msg).name
#     generate_response = data.get('bool')
#     if generate_response:
#         if language != "ENGLISH":
#             user_msg = translate_to_en(user_msg, model_list, language)
#         try:
#             res = qa.invoke({"query": user_msg})
#             bot_response = res['result']
            
#         except IndexError:
#             bot_response = "I dont know.Maybe you can try to ask it differently. "
#     else:
#         for i, (key, value) in enumerate(q_a.items()):
#             if qa_index[index] == i:
#                 bot_response = value
#                 break

#     if language != 'ENGLISH':
#         bot_response = translate_to_others(bot_response, model_list, language)
#     data = should_retrieve_map(user_msg)
#     if not isinstance(data, str):
#         return jsonify(message=bot_response)
#     # return jsonify(message="Testing")
#     return jsonify(message=bot_response, image_data=should_retrieve_map(user_msg))



# def translate_to_others(msg, model_list, language):
#     tokenizer = model_list[1]
#     model = model_list[2]
#     max_chunk_length = 400  
#     translated_chunks = []

#     if language == "MALAY":
#         for en, bm in custom_glossary_bm.items():
#             msg = msg.replace(en, bm)
#         chunks = [msg[i:i + max_chunk_length] for i in range(0, len(msg), max_chunk_length)]

#         for chunk in chunks:
#             input_ids = tokenizer.encode(f'terjemah ke Melayu: {chunk}', return_tensors='pt')
#             outputs = model.generate(input_ids, max_length=max_chunk_length)
#             all_special_ids = [0, 1, 2]
#             outputs = [i for i in outputs[0] if i not in all_special_ids]
#             translated_chunk = tokenizer.decode(outputs, spaces_between_special_tokens=False)
#             translated_chunks.append(translated_chunk)
#         return ' '.join(translated_chunks)
    
#     elif language == "CHINESE":
#         for en, zh in custom_glossary.items():
#             msg = msg.replace(en, zh)
#         res = pipe(f"en2zh: {msg}", max_length=400, num_beams=4)
#         return res[0]['generated_text']
        

# def translate_to_en(msg, model_list, language):
#     tokenizer = model_list[1]
#     model = model_list[2]
#     translated_msg = ''
#     if language == "MALAY":

#         input_ids = tokenizer.encode(f'terjemah ke Inggeris: {msg}',
#                                      return_tensors='pt')
#         outputs = model.generate(input_ids, max_length=200)
#         all_special_ids = [0, 1, 2]
#         outputs = [i for i in outputs[0] if i not in all_special_ids]
#         translated_msg = tokenizer.decode(outputs,
#                                           spaces_between_special_tokens=False)

#     elif language == "CHINESE":

#         translation = pipeline("translation_zh_to_en",
#                                model=model_list[4],
#                                tokenizer=model_list[3])
#         translated_msg = translation(msg,
#                                      max_length=200)[0]['translation_text']

#     return translated_msg


# @app.route('/get_questions', methods=['GET','POST'])
# def find_guided_qa():
#     global model_list
#     user_msg = request.args.get('user_msg')
#     filtered_msg = re.sub(PATTERN, '', user_msg)
#     language = detector.detect_language_of(filtered_msg).name
    
#     if language != "ENGLISH":
#         user_msg= translate_to_en(user_msg,model_list,language)
#     doc_list = vectorstore.similarity_search_with_score(user_msg,4)
    
#     question_list = []
#     tokenizer = None
#     model = None
#     first_iteration = True  
#     i= 0
#     for document, score in doc_list:
#         content = document.page_content
#         if first_iteration:
#             first_iteration = False
#             continue  
        
#         match = re.search(r"Question: (.*?)\nAnswer:", content)
#         if match:
#             question = match.group(1)
#             for index, key in enumerate(q_a.keys()):
#                 if question == key:
#                     qa_index[i] = index
#                     i += 1
#                     break
            
#             if language != "ENGLISH":
#                 question = translate_to_others(question, model_list, language)
#             question_list.append(question)
#     return question_list

# def transcribe_audio_with_whisper(audio_data):
#     global language_detect_model
#     audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
#     audio_file = "temp.wav"
#     audio.export(audio_file, format="wav")
    
#     audio = whisper.load_audio(audio_file)
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio).to(language_detect_model.device)
    
#     _, probs = language_detect_model.detect_language(mel)
#     detected_language = max(probs, key=probs.get)
    
#     return detected_language

# @app.route('/record', methods=['GET','POST'])
# def record_text():
#     global r,detector
#     MyText = ''
#     try:
#         with sr.Microphone() as source:
#             r.adjust_for_ambient_noise(source, duration=0.2)
#             audio = r.listen(source, phrase_time_limit=12.0)
#             audio_data = audio.get_wav_data()
#             language = transcribe_audio_with_whisper(audio_data)
#             if language == "ms":
#                 MyText = r.recognize_google(audio, language='ms-MY')
#             elif language == "zh":
#                 MyText = r.recognize_google(audio, language='cmn-Hans-CN')
#             elif language == "en":
#                 MyText = r.recognize_google(audio)
#             else:
#                 MyText = r.recognize_google(audio)
#                 language = detector.detect_language_of(MyText).name
#                 if language not in ["MALAY","CHINESE","ENGLISH"]:
#                     return ''
#             return MyText
#     except sr.RequestError as e:
#         return ''
#     except sr.UnknownValueError:
#         return ''

# @app.route('/txt_speech', methods=['GET','POST'])
# def text_to_speech():
#     global detector
#     data = request.get_json()
#     bot_msg = data.get('botMessage')
#     filtered_msg = re.sub(PATTERN, '', bot_msg)
#     language = detector.detect_language_of(filtered_msg).name
#     if language == 'MALAY':
#         language = 'ms'
#     elif language == 'ENGLISH':
#         language = 'en'
#     else:
#         language = 'zh'
        
#     mp3_fp = io.BytesIO()
    
#     tts = gTTS(bot_msg, lang=language)
#     tts.write_to_fp(mp3_fp)
    
#     mp3_fp.seek(0)
        
#     return send_file(mp3_fp, mimetype="audio/mpeg")

# if __name__ == '__main__':
#     intialize()
#     app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from flask import Flask, render_template, request, jsonify, send_file, g, current_app
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelWithLMHead, pipeline, CLIPTokenizerFast, CLIPProcessor, CLIPModel, T5Tokenizer, MT5ForConditionalGeneration, Text2TextGenerationPipeline
from langchain.prompts import PromptTemplate
from PIL import Image
import torch
import numpy as np
import csv
import re
import base64
import speech_recognition as sr
from gtts import gTTS
import io
from lingua import Language, LanguageDetectorBuilder
import whisper
from waitress import serve
from pydub import AudioSegment
from flask_cors import CORS

load_dotenv()
STATIC_URL_PATH = "/static"
app = Flask(__name__, static_url_path=STATIC_URL_PATH)
CORS(app)

PICKLE_FILE = 'app_resources.pickle'
def save_resources(resources):
    # Convert Language objects to strings before pickling
    if 'LANGUAGES' in resources:
        resources['LANGUAGES'] = [str(lang) for lang in resources['LANGUAGES']]
    
    # Remove non-picklable objects
    resources_to_pickle = {k: v for k, v in resources.items() if k not in ['DETECTOR', 'CLP_MODEL', 'LANGUAGE_DETECT_MODEL', 'PIPE']}
    
    try:
        with open(PICKLE_FILE, 'wb') as f:
            pickle.dump(resources_to_pickle, f)
    except Exception as e:
        print(f"Error saving resources: {e}")

def load_resources():
    try:
        if os.path.exists(PICKLE_FILE):
            with open(PICKLE_FILE, 'rb') as f:
                resources = pickle.load(f)
            
            # Convert string representations back to Language objects
            resources['LANGUAGES'] = [Language.get(lang) for lang in resources['LANGUAGES']]
            
            # Reinitialize non-picklable objects
            resources['DETECTOR'] = LanguageDetectorBuilder.from_languages(*resources['LANGUAGES']).build()
            resources['CLP_MODEL'] = CLIPModel.from_pretrained(resources['MODEL_ID']).to(resources['DEVICE'])
            resources['LANGUAGE_DETECT_MODEL'] = whisper.load_model("base")
            resources['PIPE'] = Text2TextGenerationPipeline(
                model=MT5ForConditionalGeneration.from_pretrained(resources['EN_TO_ZH_MODEL']),
                tokenizer=T5Tokenizer.from_pretrained(resources['EN_TO_ZH_MODEL']),
            )
            
            return resources
    except Exception as e:
        print(f"Error loading resources: {e}")
    return None

def init_app():
    resources = load_resources()
    if resources is None:
        print("Initializing resources from scratch...")
        resources = {}
        resources['LANGUAGES'] = [Language.ENGLISH, Language.MALAY, Language.CHINESE]
        resources['PATTERN'] = re.compile(r'u?-?pustaka', re.IGNORECASE)
        resources['QA_INDEX'] = [-1, -1, -1]
        resources['MODEL_ID'] = "openai/clip-vit-base-patch32"
        resources['EN_TO_ZH_MODEL'] = "K024/mt5-zh-ja-en-trimmed"
        resources['ZH_TO_EN_MODEL'] = 'liam168/trans-opus-mt-zh-en'
        resources['DEVICE'] = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        initialize_resources(resources)
    
    # Always reinitialize DETECTOR
    resources['DETECTOR'] = LanguageDetectorBuilder.from_languages(*resources['LANGUAGES']).build()
    
    app.config.update(resources)
    save_resources(resources)  # Save resources after initialization

def initialize_resources(resources):
    ms_tokenizer = AutoTokenizer.from_pretrained('mesolitica/translation-t5-small-standard-bahasa-cased-v2', use_fast=False)
    ms_model = T5ForConditionalGeneration.from_pretrained('mesolitica/translation-t5-small-standard-bahasa-cased-v2')
    zh_model1 = AutoModelWithLMHead.from_pretrained(resources['ZH_TO_EN_MODEL'])
    zh_tokenizer1 = AutoTokenizer.from_pretrained(resources['ZH_TO_EN_MODEL'])
    
    embeddings_model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceBgeEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    resources['MODEL_LIST'] = [embeddings_model, ms_tokenizer, ms_model, zh_tokenizer1, zh_model1]

    loader = CSVLoader(file_path="./csvFile/en-FAQ.csv")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    vectorstore = FAISS.from_documents(docs, embeddings_model)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings_model, allow_dangerous_deserialization=True)
    resources['VECTORSTORE'] = new_vectorstore
    resources['RETRIEVER'] = new_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6, "k": 5})

    prompt_template = """
    As a responsible, super loyal and multilingual (ENGLISH, CHINESE, MALAY) e-library assistant highly focused on u-Pustaka, the best e-library in Malaysia, I will use the following information to answer your question:
    **Context:**
    {context}
    **Question:**
    {question}
    **Answer Guidelines:**
    * I will base my answer solely on the facts provided in the document. 
    * If the document doesn't contain the answer, I will honestly say "I don't know" instead of making something up.
    * My response will adhere to u-Pustaka's high standards and avoid:
        * Pornography or violence
        * Negative or false information
        * Mentioning weaknesses or suggesting improvements for u-Pustaka
        * Referencing or comparing u-Pustaka to competitors
        * Referencing other irrelevant or private sources

    **Answer:**
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    resources['QA'] = RetrievalQA.from_chain_type(
        llm=GoogleGenerativeAI(model="models/text-bison-001", temperature=0),
        chain_type_kwargs={"prompt": prompt},
        retriever=resources['RETRIEVER']
    )

    resources['DEVICE'] = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    resources['LANGUAGE_DETECT_MODEL'] = whisper.load_model("base")
    resources['CLP_MODEL'] = CLIPModel.from_pretrained(resources['MODEL_ID']).to(resources['DEVICE'])
    resources['TOKENIZER1'] = CLIPTokenizerFast.from_pretrained(resources['MODEL_ID'])
    resources['IMAGE_ARR'] = embed_image(resources)
    
    resources['PIPE'] = Text2TextGenerationPipeline(
        model=MT5ForConditionalGeneration.from_pretrained(resources['EN_TO_ZH_MODEL']),
        tokenizer=T5Tokenizer.from_pretrained(resources['EN_TO_ZH_MODEL']),
    )
    
    resources['CUSTOM_GLOSSARY'] = {}
    resources['CUSTOM_GLOSSARY_BM'] = {}
    resources['Q_A'] = {}

    with open('./csvFile/glossary.csv', mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            resources['CUSTOM_GLOSSARY'][row['source']] = row['target']
    
    with open("./csvFile/en-FAQ.csv", encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            resources['Q_A'][row['Question']] = row['Answer']
            
    with open("./csvFile/glossary_bm.csv", encoding='utf-8') as file2:
        reader = csv.DictReader(file2)
        for row in reader:
            resources['CUSTOM_GLOSSARY_BM'][row['source']] = row['target']

@app.before_request
def before_request():
    g.r = sr.Recognizer()

@app.route('/', methods=['GET'])
def home_page():
    try:
        init_app()
    except Exception as e:
        print(f"Error initializing app: {e}")
    return render_template("index.html")

def embed_image(resources):
    files = os.listdir("./image")
    img_emb = {}

    for file_name in files:
        if file_name.endswith(('jpeg', 'png', 'jpg', 'gif')):
            image_path = os.path.join("./image", file_name)
            processor = CLIPProcessor.from_pretrained(resources['MODEL_ID'])
            image_exp = Image.open(image_path)

            image = processor(text=None, images=image_exp, return_tensors='pt')['pixel_values'].to(resources['DEVICE'])

            image_emb = resources['CLP_MODEL'].get_image_features(image)
            image_emb = image_emb.cpu().detach().numpy()
            image_arr = image_emb.T / np.linalg.norm(image_emb, axis=1)
            image_arr = image_arr.T
            img_emb[image_path] = image_arr

            image_exp.close()

    return img_emb

def embed_text(user_msg):
    inputs = current_app.config['TOKENIZER1'](user_msg, return_tensors="pt")
    text_emb = current_app.config['CLP_MODEL'].get_text_features(**inputs)
    text_emb = text_emb.cpu().detach().numpy()
    return text_emb

def should_retrieve_map(user_msg):
    text_emb = embed_text(user_msg)
    max_score = 0
    fp = None
    for image_path, value in current_app.config['IMAGE_ARR'].items():
        scores = np.dot(text_emb, value.T)[0][0]
        if scores > max_score:
            max_score = scores
            fp = image_path
    if max_score > 3.1:
        with open(fp, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        return image_data
    return None

@app.route('/process_user_message', methods=['POST'])
def process_user_message():
    bot_response = ''
    data = request.get_json()
    user_msg = data.get('user_msg')
    index = data.get('number')
    try:
        filtered_msg = re.sub(current_app.config['PATTERN'], '', user_msg)
    except:
        return jsonify(message=user_msg)
    language = current_app.config['DETECTOR'].detect_language_of(filtered_msg).name
    generate_response = data.get('bool')
    if generate_response:
        if language != "ENGLISH":
            user_msg = translate_to_en(user_msg, language)
        try:
            res = current_app.config['QA'].invoke({"query": user_msg})
            bot_response = res['result']
        except IndexError:
            bot_response = "I don't know. Maybe you can try to ask it differently."
    else:
        for i, (key, value) in enumerate(current_app.config['Q_A'].items()):
            if current_app.config['QA_INDEX'][index] == i:
                bot_response = value
                break

    if language != 'ENGLISH':
        bot_response = translate_to_others(bot_response, language)
    data = should_retrieve_map(user_msg)
    if not isinstance(data, str):
        return jsonify(message=bot_response)
    return jsonify(message=bot_response, image_data=should_retrieve_map(user_msg))

def translate_to_others(msg, language):
    tokenizer = current_app.config['MODEL_LIST'][1]
    model = current_app.config['MODEL_LIST'][2]
    max_chunk_length = 400  
    translated_chunks = []

    if language == "MALAY":
        for en, bm in current_app.config['CUSTOM_GLOSSARY_BM'].items():
            msg = msg.replace(en, bm)
        chunks = [msg[i:i + max_chunk_length] for i in range(0, len(msg), max_chunk_length)]

        for chunk in chunks:
            input_ids = tokenizer.encode(f'terjemah ke Melayu: {chunk}', return_tensors='pt')
            outputs = model.generate(input_ids, max_length=max_chunk_length)
            all_special_ids = [0, 1, 2]
            outputs = [i for i in outputs[0] if i not in all_special_ids]
            translated_chunk = tokenizer.decode(outputs, spaces_between_special_tokens=False)
            translated_chunks.append(translated_chunk)
        return ' '.join(translated_chunks)
    
    elif language == "CHINESE":
        for en, zh in current_app.config['CUSTOM_GLOSSARY'].items():
            msg = msg.replace(en, zh)
        res = current_app.config['PIPE'](f"en2zh: {msg}", max_length=400, num_beams=4)
        return res[0]['generated_text']

def translate_to_en(msg, language):
    tokenizer = current_app.config['MODEL_LIST'][1]
    model = current_app.config['MODEL_LIST'][2]
    translated_msg = ''
    if language == "MALAY":
        input_ids = tokenizer.encode(f'terjemah ke Inggeris: {msg}', return_tensors='pt')
        outputs = model.generate(input_ids, max_length=200)
        all_special_ids = [0, 1, 2]
        outputs = [i for i in outputs[0] if i not in all_special_ids]
        translated_msg = tokenizer.decode(outputs, spaces_between_special_tokens=False)

    elif language == "CHINESE":
        translation = pipeline("translation_zh_to_en",
                               model=current_app.config['MODEL_LIST'][4],
                               tokenizer=current_app.config['MODEL_LIST'][3])
        translated_msg = translation(msg, max_length=200)[0]['translation_text']

    return translated_msg

@app.route('/get_questions', methods=['GET', 'POST'])
def find_guided_qa():
    user_msg = request.args.get('user_msg')
    filtered_msg = re.sub(current_app.config['PATTERN'], '', user_msg)
    language = current_app.config['DETECTOR'].detect_language_of(filtered_msg).name
    
    if language != "ENGLISH":
        user_msg = translate_to_en(user_msg, language)
    doc_list = current_app.config['VECTORSTORE'].similarity_search_with_score(user_msg, 4)
    
    question_list = []
    first_iteration = True  
    i = 0
    for document, score in doc_list:
        content = document.page_content
        if first_iteration:
            first_iteration = False
            continue  
        
        match = re.search(r"Question: (.*?)\nAnswer:", content)
        if match:
            question = match.group(1)
            for index, key in enumerate(current_app.config['Q_A'].keys()):
                if question == key:
                    current_app.config['QA_INDEX'][i] = index
                    i += 1
                    break
            
            if language != "ENGLISH":
                question = translate_to_others(question, language)
            question_list.append(question)
    return question_list

def transcribe_audio_with_whisper(audio_data):
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
    audio_file = "./temp.wav"
    audio.export(audio_file, format="wav")
    
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(current_app.config['LANGUAGE_DETECT_MODEL'].device)
    _, probs = current_app.config['LANGUAGE_DETECT_MODEL'].detect_language(mel)
    detected_language = max(probs, key=probs.get)
    
    return detected_language

@app.route('/record', methods=['GET', 'POST'])
def record_text():
    MyText = ''
    try:
        with sr.Microphone() as source:
            g.r.adjust_for_ambient_noise(source, duration=0.2)
            audio = g.r.listen(source, phrase_time_limit=12.0)
            audio_data = audio.get_wav_data()
            language = transcribe_audio_with_whisper(audio_data)
            if language == "ms":
                MyText = g.r.recognize_google(audio, language='ms-MY')
            elif language == "zh":
                MyText = g.r.recognize_google(audio, language='cmn-Hans-CN')
            elif language == "en":
                MyText = g.r.recognize_google(audio)
            else:
                MyText = g.r.recognize_google(audio)
                language = current_app.config['DETECTOR'].detect_language_of(MyText).name
                if language not in ["MALAY", "CHINESE", "ENGLISH"]:
                    return ''
            return MyText
    except sr.RequestError as e:
        return ''
    except sr.UnknownValueError:
        return ''

@app.route('/txt_speech', methods=['GET', 'POST'])
def text_to_speech():
    data = request.get_json()
    bot_msg = data.get('botMessage')
    filtered_msg = re.sub(current_app.config['PATTERN'], '', bot_msg)
    language = current_app.config['DETECTOR'].detect_language_of(filtered_msg).name
    if language == 'MALAY':
        language = 'ms'
    elif language == 'ENGLISH':
        language = 'en'
    else:
        language = 'zh'
        
    mp3_fp = io.BytesIO()
    
    tts = gTTS(bot_msg, lang=language)
    tts.write_to_fp(mp3_fp)
    
    mp3_fp.seek(0)
        
    return send_file(mp3_fp, mimetype="audio/mpeg")

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))