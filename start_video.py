# %%


# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pickle
import pandas as pd
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import random
# %%
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

# Import for voice
from threading import Thread
import speech_recognition as sr
import time

#data stracture to hold context

context={}

# importing vlc module 
import vlc 
# importing time module 
import time 
# creating vlc media player object 
media_player = vlc.MediaPlayer() 
# setting full screen status 
media_player.set_fullscreen(True) 




# Hotfix function
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

# Run the function
make_keras_picklable()




# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)
    
data = pickle.load( open( "katana-assistant-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']

# %%
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

# %%

# %%
# Use pickle to load in the pre-trained model
global graph
graph = tf.get_default_graph()

with open(f'katana-assistant-model.pkl', 'rb') as f:
    model = pickle.load(f)

# %%
def classify(sentence):
    ERROR_THRESHOLD = 0.45
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    
    return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    #print(results[0][0])
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        #return random.choice(i['responses'])
                        #return results[0][0]
                    return results[0][0]

            results.pop(0)

def vid_play(vid_path):
    path="video/"+vid_path+".mp4"
    # media object 
    media = vlc.Media(path) 
    # setting media to the media player 
    media_player.set_media(media) 
    # start playing video 
    media_player.play() 
    time.sleep(1.5)
    duration = media_player.get_length() / 1000
    time.sleep(duration)
    
#main program to run

#Welcome note
vid_play("welcome")



#welcome message play welcome.mp4


res_p='0'
res='0'

flag_nr=0
#loop till there exist a face
while(True):
    print(" now ASK YOUR next QUESTION")
       
    if flag_nr==0:
    #play ny_quary.mp4
        vid_play("ny_quary")
       
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration = 1)
        try:
            audio = r.listen(source, timeout = 5)
            text = r.recognize_google(audio)
            print("You said : {}".format(text))

            rsp=response(str(text))     
            print('Here is your answer:',rsp)
            vid_play(rsp)
            flag_nr=0  
        except:
            time.sleep(1)
            print("Sorry could not recognize what you said")
            vid_play("not_rec")
            flag_nr=1
            
 
# %%


