from django.http import HttpResponse
import boto3
from django.conf import settings
import torch
import torch.onnx
import PyPDF2 as pdf
from transformers import BertForQuestionAnswering , BartTokenizer ,AutoModelForQuestionAnswering ,BartForConditionalGeneration ,AutoTokenizer, AutoModelWithLMHead, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering
from io import BytesIO
import json
from django.views.decorators.csrf import csrf_exempt
import csv
import nltk
from docx import Document
from bs4 import BeautifulSoup
import requests
import re
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.tokenize import sent_tokenize

''' I am using DRF (Django Rest Framework for this)'''

model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
optimum_qa = pipeline("question-answering", model=model, tokenizer=tokenizer, max_length=512)
max_length = model.config.max_position_embeddings  # Account for [CLS] and [SEP] tokens

# Create your views here.
def clean_context(text):
    input_text=re.sub(r'\s+'," ",text)
    input_text = re.sub(r'\[[^\]]*\]', '', input_text)
    text_with_line_breaks = re.sub(r'([!?])', r'\1\n', input_text)
    return text_with_line_breaks

def genrate_summary(text,max_chunk_length,max_summary_length):
    print('In genrate_summary function:--->')
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    input_chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    # Generate summaries for each chunk
    summaries = []
    for chunk in input_chunks:
        inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=max_chunk_length, truncation=True)
        summary_ids = model.generate(inputs, num_beams=4, min_length=30, max_length=max_summary_length, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    # Combine the summaries to form the final output
    final_summary = " ".join(summaries)
    return final_summary

    
@csrf_exempt
def getting_details(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        s3 = boto3.client('s3', aws_access_key_id=settings.AWS_S3_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_S3_SECRET_ACCESS_KEY)
        file_response = s3.get_object(Bucket='dl-chat-bucket',Key='Samplequestions.pdf')
        file_extension = file_response['ContentType']
        pdf_data = file_response['Body'].read()
        if file_extension == 'text/plain':
            text = pdf_data.decode('utf-8')
        elif file_extension == 'text/csv':
            text = ""
            csv_data = pdf_data.decode('utf-8').splitlines()
            reader = csv.reader(csv_data)
            for row in reader:
                text += ','.join(row) + '\n'
        elif file_extension == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            document = Document(BytesIO(pdf_data))
            text = '\n'.join([p.text for p in document.paragraphs])
        
        elif file_extension == 'application/pdf':
            pdf_reader = pdf.PdfReader(BytesIO(pdf_data))
            num_pages = len(pdf_reader.pages)
            text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
                extracted_text=text.lower()
        else:
            text = "Unsupported file type"
        extracted_text=text.lower()
        cleaned_text = clean_context(extracted_text)
        try:
            summarized_text=genrate_summary(cleaned_text ,max_chunk_length=512,max_summary_length=512)
            print(summarized_text,'-------------------> summarized Text from summarizer')
        except Exception as e:
            print(e,"=> Error occurs due to this ")

        answer=optimum_qa(question,summarized_text)

        print("Answer------------>>>>",answer)

        response_data = {'answer': answer['answer']}
        return HttpResponse(json.dumps(response_data), content_type='application/json')


