from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import pegas_1

parser = argparse.ArgumentParser() #create object
parser.add_argument("--news_videoid", help='path of new article',default = '3Rzi11Hvyh0')
parser.add_argument("--model_file",help='file path of the Pegasus model',default = 'model/')
parser.add_argument('--sp_model_file', help = 'sentence piece model',
default =  'ckpt/c4.unigram.newline.10pct.96000.model')
args = parser.parse_args()

transcript_list = YouTubeTranscriptApi.list_transcript(args.news_videoid)

text = ''
for transcript in transcript_list:
    if transcript.language_code == 'en':
        data = transcript.fetch()
        for sentence in data:
            text += sentence['text']+' '
        
encoder = pegas_1.create_encoder(args.sp_model_file)
            
# adjust input size to 1024 (to fit neural network)
ids = encoder.tokenize(text) #pad 0 无字符 1024只能
inputs = np.zeros(1024)
input_length = len(ids)
if input_length > 1024: input_length = 1024
inputs[:input_length] = ids[:input_length]
            
# load Pegasus model from file
imported = tf.saved_model.load(args.model_file,tags = "serve")
            
#set up inputs
example = tf.train.Example()
example.features.feature['inputs'].int64_list.value.extend(inputs.astype(int))
            
#generate output
output = imported.signatures['serving_default'](examples=tf.constant([example.SerializeToString()]))
            
#detokenize results
summarization = encoder.detokenize(output['outputs'].numpy().flatten().tolist())
print("The summarization of the article is:"+summarization)
