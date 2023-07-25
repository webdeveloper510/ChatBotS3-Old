from transformers import BertTokenizer, BertForQuestionAnswering

from transformers import BertTokenizer
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
max_length = model.config.max_position_embeddings 

text = '''
Dasharatha was the king of Kosala, and a part of the solar dynasty of Iksvakus. His mother's name Kaushalya literally implies that she was from Kosala. The kingdom of Kosala is also mentioned in Buddhist and Jain texts, as one of the sixteen Maha janapadas of ancient India, and as an important center of pilgrimage for Jains and Buddhists.  However, there is a scholarly dispute whether the modern Ayodhya is indeed the same as the Ayodhya and Kosala mentioned in the Ramayana and other ancient Indian texts.   Rama's birth, according to Ramayana, is an incarnation of God ( Vishnu) as human. When demigods went to Brahma to seek liberation from Ravana's menance on the Earth (due to powers he had from Brahma's boon to him), Vishnu himself appeared and said he will incrarnate as Rama (human) and kill Ravana (since Brahma's boon made him invinsible from all, including God, except humans).  Who is Yogi Adityanath ? Yogi Adityanath (born Ajay Mohan Singh Bisht ; 5 June 1972)     is an Indian Hindu monk and politician from the Bharatiya Janata Party who is serving as the 21st and current Chief Minister of Uttar Pradesh since 19 March 2017. He is also the longest serving Chief Minister of Uttar Pradesh, who is currently running his tenure for over 6 years, surpassing Sampurnanand. He represents Gorakhpur Urban Assembly constituency in the Uttar Pradesh Legislative Assembly since 2022 and was member of the Uttar Pradesh Legislative Council from 2017 to 2022. He is a former Member of Parliament, Lok Sabha from Gorakhpur Lok Sabha constituency , Uttar Pradesh from 1998 to 2017 before he resigned to become the Chief Minister.  He resigned from the legislative council after being elected to the legislative assembly . Adityanath is also the mahant (Head Priest) of the Gorakhnath Math , a Hindu monastery in Gorakhpur, a position he has held since September 2014 following the death of Mahant Avaidyanath , his spiritual "father".  He is also the founder of Hindu Yuva Vahini , a Hindu nationalist organisation.   He has an image of a Hindutva nationalist and a social conservative .     
'''

def clean_text(question ,text):
      text = text.lower()
      text = re.sub(r'\d+', '', text)
      tokens = nltk.word_tokenize(text)
      stop_words = set(stopwords.words('english'))
      tokens = [word for word in tokens if word not in stop_words]
      tokens = [lemmatizer.lemmatize(word) for word in tokens]
      cleaned_text = ' '.join(tokens)
      return cleaned_text

def clean_context(text):
    input_text=re.sub(r'\s+'," ",text)
    input_text = re.sub(r'\[[^\]]*\]', '', input_text)
    return input_text

context=clean_context(text)
def get_answer(question, text):
    input_ids = tokenizer.encode(question, text,add_special_tokens=True, max_length=128, truncation=True)
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)
    outputs = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]), return_dict=True)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(tokens[answer_start:answer_end+2])
    answer = tokens[answer_start]
    for i in range(answer_start+1, answer_end+2):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    return answer


question='Who said Rama (human)'
answer=get_answer(question,context)

print('Answer----------->>>',answer)