import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelWithLMHead, pipeline, CLIPTokenizerFast, CLIPProcessor, CLIPModel
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
import numpy as np
import csv
import re
import base64
import speech_recognition as sr

prompt_template = """
As a responsible and super loyal e-library assistant highly focused on u-Pustaka, the best e-library in Malaysia, I will use the following information to answer your question:

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
    * Referencing other irrelevent or private sources
    

**Answer:**

"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])

MODEL_ID = "openai/clip-vit-base-patch32"
en_to_zh_model = 'liam168/trans-opus-mt-en-zh'
zh_to_en_model = 'liam168/trans-opus-mt-zh-en'



load_dotenv()
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("main.html")


def csv_to_dict(file, encoding='utf-8'):
    en_dict = {}

    try:
        with open(file, encoding=encoding) as enFAQ:

            reader = csv.DictReader(enFAQ)

            for row in reader:

                key = row['Question']
                value = row['Answer']

                en_dict[key] = value
    except FileNotFoundError:

        print("File does not exist")

    return en_dict


def load_modal():

    ms_tokenizer = AutoTokenizer.from_pretrained(
        'mesolitica/translation-t5-small-standard-bahasa-cased-v2',
        use_fast=False)
    ms_model = T5ForConditionalGeneration.from_pretrained(
        'mesolitica/translation-t5-small-standard-bahasa-cased-v2')
    zh_model = AutoModelWithLMHead.from_pretrained(en_to_zh_model)
    zh_tokenizer = AutoTokenizer.from_pretrained(en_to_zh_model)
    zh_model1 = AutoModelWithLMHead.from_pretrained(zh_to_en_model)
    zh_tokenizer1 = AutoTokenizer.from_pretrained(zh_to_en_model)
    embeddings_model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=embeddings_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs)

    return [
        embeddings_model, ms_tokenizer, ms_model, zh_tokenizer, zh_model,
        zh_tokenizer1, zh_model1
    ]


def process_doc(embeddings):

    loader = CSVLoader(file_path="en-FAQ.csv")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=30,
                                          separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    with open("question.txt", "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    return vectorstore


def embed_image():

    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    image_exp = Image.open("image/National_Library_Map.jpg")

    image = processor(text=None, images=image_exp,
                      return_tensors='pt')['pixel_values'].to(device)

    image_emb = Clp_model.get_image_features(image)
    image_emb = image_emb.cpu().detach().numpy()
    image_arr = image_emb.T / np.linalg.norm(image_emb, axis=1)
    image_arr = image_arr.T
    return image_arr


def embed_text(user_msg):

    inputs = tokenizer1(user_msg, return_tensors="pt")
    text_emb = Clp_model.get_text_features(**inputs)
    text_emb = text_emb.cpu().detach().numpy()
    return text_emb

def should_retrieve_map(user_msg):
    text_emb = embed_text(user_msg)
    scores = np.dot(text_emb, image_arr.T)
    print(f"scores: {scores}")
    if scores > 3.1:
        image_path = "image/National_Library_Map.jpg"
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        return image_data
    return None


@app.route('/process_user_message', methods=['POST'])
def process_user_message():

    bot_response = ''
    data = request.get_json()
    user_msg = data.get('string')
    language = data.get('language')

    if language != "eng":
        user_msg = translate_to_en(user_msg, model_list, language)
        print(f"Other languages user message: {user_msg}")
    try:
        res = qa.invoke({"query": user_msg})
        bot_response = res['result']
    except IndexError:
        bot_response = "I dont know.Maybe you can try to ask it differently. "
        
    print(bot_response)
    if language != 'eng':
        bot_response = translate_to_others(bot_response, model_list, language)
        print(f"Other languages bot response: {bot_response}")

    data = should_retrieve_map(user_msg)
    if not isinstance(data, str):
        return jsonify(message=bot_response)
    return jsonify(message=bot_response, image_data=should_retrieve_map(user_msg))
    


def translate_to_others(msg, model_list, language):
    tokenizer = model_list[1]
    model = model_list[2]
    max_chunk_length = 400  
    translated_chunks = []

    if language == "bm":
        chunks = [msg[i:i + max_chunk_length] for i in range(0, len(msg), max_chunk_length)]
        
        for chunk in chunks:
            input_ids = tokenizer.encode(f'terjemah ke Melayu: {chunk}', return_tensors='pt')
            outputs = model.generate(input_ids, max_length=max_chunk_length)
            all_special_ids = [0, 1, 2]
            outputs = [i for i in outputs[0] if i not in all_special_ids]
            translated_chunk = tokenizer.decode(outputs, spaces_between_special_tokens=False)
            translated_chunks.append(translated_chunk)
    
    elif language == "cn":
        translation = pipeline("translation_en_to_zh", model=model_list[4], tokenizer=model_list[3])
        chunks = [msg[i:i + max_chunk_length] for i in range(0, len(msg), max_chunk_length)]
        
        for chunk in chunks:
            translated_chunk = translation(chunk, max_length=max_chunk_length)[0]['translation_text']
            translated_chunks.append(translated_chunk)

    translated_msg = ' '.join(translated_chunks)
    return translated_msg.strip()


def translate_to_en(msg, model_list, language):
    tokenizer = model_list[1]
    model = model_list[2]
    translated_msg = ''
    if language == "bm":

        input_ids = tokenizer.encode(f'terjemah ke Inggeris: {msg}',
                                     return_tensors='pt')
        outputs = model.generate(input_ids, max_length=200)
        all_special_ids = [0, 1, 2]
        outputs = [i for i in outputs[0] if i not in all_special_ids]
        translated_msg = tokenizer.decode(outputs,
                                          spaces_between_special_tokens=False)

    elif language == "cn":

        translation = pipeline("translation_zh_to_en",
                               model=model_list[6],
                               tokenizer=model_list[5])
        translated_msg = translation(msg,
                                     max_length=200)[0]['translation_text']

    return translated_msg


@app.route('/get_questions', methods=['GET'])
def find_guided_qa():

    user_msg = request.args.get('user_msg')
    language = request.args.get('language')

    doc_list = vectorstore.similarity_search(user_msg, 4)

    question_list = []
    tokenizer = None
    model = None
    counter = 1
    for question in doc_list:
        if counter == 1:
            counter += 1
            continue
        match = re.search(r"Question: (.*?)\nAnswer:", question.page_content)
        if match:
            question = match.group(1)
            if language != "eng":
                question = translate_to_others(question, model_list, language)
            question_list.append(question)

    return question_list


@app.route('/record', methods=['POST'])
def record_text():
    data = request.get_json()
    language = data.get('language')
    print(language)
    MyText = ''
    try:
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.2)
            audio2 = r.listen(source2, phrase_time_limit=12.0)
            if language == "bm":
                MyText = r.recognize_google(audio2, language='ms-MY')
            elif language == "cn":
                MyText = r.recognize_google(audio2, language='cmn-Hans-CN')
            else:
                MyText = r.recognize_google(audio2)
                
            return MyText
    except sr.RequestError as e:
        print("Could not request results: {0}".format(e))
    except sr.UnknownValueError:
        print("Unknown error occurred")


if __name__ == '__main__':
    
    data_dict = csv_to_dict("en-FAQ.csv")
    model_list = load_modal()
    vectorstore = process_doc(model_list[0])
    new_vectorstore = FAISS.load_local("faiss_index_react",
                                       model_list[0],
                                       allow_dangerous_deserialization=True)
    retriever = new_vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.6,
            "k": 5
        })
    qa = RetrievalQA.from_chain_type(llm=GoogleGenerativeAI(
        model="models/text-bison-001", temperature=0),
                                     chain_type_kwargs={"prompt": prompt},
                                     retriever=retriever)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    Clp_model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    tokenizer1 = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    image_arr = embed_image()
    r = sr.Recognizer()
    app.run(debug=True)
