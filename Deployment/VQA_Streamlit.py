import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from PIL import Image
import pickle
import os, urllib, cv2, re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title('Visual Question Answering')

def main():
    
    st.sidebar.title("Project Content")
#     app_mode = st.sidebar.selectbox("Choose the app mode",
#         ["1. Project Brief", "2. Data Info", "3. Test VQA"])
#     if app_mode == "1. Project Brief":
#         st.sidebar.success('To continue select "Load data".')
#         instructions()
#     elif app_mode == "2. Data Info":
#         st.sidebar.success('To continue select "Make a prediction".')
#         load_data()
#     elif app_mode == "3. Test VQA":
#         make_prediction()

    option = st.sidebar.radio("Select content",("I. Project Brief", "II. Data Info", "III. Test VQA"))

    if option == 'I. Project Brief':
        brief()  
    elif option == 'II. Data Info':
        load_data()
    elif option == 'III. Test VQA':
        make_prediction()

def brief():
    st.write("This project demonstrates the Visual Question Answering dataset (https://visualqa.org/download.html) and deeper LSTM + pre-trained VGG19 model (https://arxiv.org/pdf/1505.00468.pdf) into an interactive Streamlit (https://streamlit.io) app.")
    st.write("In the VQA task, a free-form natural language question about an image and that image are taken as input, and give a natural language short answer as the output.")
    st.write("**Please select next content in the sidebar to check Data Info & Test VQA.**")
    st.write("For more information, Check out this blog (https://medium.com/@niralidedaniya)")
    
def load_data():
    st.text('Loading Data...')
    data = pd.read_csv('mscoco_train2014_k1000_50k.csv')
    st.text('Loading Data....Done!')
    st.subheader('Raw Data')
    st.write(data)
    
    st.subheader('Analysis of Questions')
    ans_df = pd.DataFrame({
        'Questions':(data['que_firstword'].value_counts()[:20]).index, 
        'Counts':(data['que_firstword'].value_counts()[:20]).values})
    c = alt.Chart(ans_df).mark_bar().encode(x=alt.X('Questions',sort=None), y='Counts')
    st.altair_chart(c, use_container_width=True)
    
    st.subheader('Analysis of Answers')
    ans_df = pd.DataFrame({
        'Answers':(data['answer'].value_counts()[:20]).index, 
        'Counts':(data['answer'].value_counts()[:20]).values})
    c = alt.Chart(ans_df).mark_bar().encode(x=alt.X('Answers',sort=None), y='Counts')
    st.altair_chart(c, use_container_width=True)
    

def make_prediction():
    
    def decontractions(phrase):
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"won\’t", "will not", phrase)
        phrase = re.sub(r"can\’t", "can not", phrase)
        phrase = re.sub(r"he\'s", "he is", phrase)
        phrase = re.sub(r"she\'s", "she is", phrase)
        phrase = re.sub(r"it\'s", "it is", phrase)
        phrase = re.sub(r"he\’s", "he is", phrase)
        phrase = re.sub(r"she\’s", "she is", phrase)
        phrase = re.sub(r"it\’s", "it is", phrase)
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"n\’t", " not", phrase)
        phrase = re.sub(r"\’re", " are", phrase)
        phrase = re.sub(r"\’d", " would", phrase)
        phrase = re.sub(r"\’ll", " will", phrase)
        phrase = re.sub(r"\’t", " not", phrase)
        phrase = re.sub(r"\’ve", " have", phrase)
        phrase = re.sub(r"\’m", " am", phrase)
        return phrase

    def text_preprocess(text):
        text = text.lower()
        text = decontractions(text) # replace contractions into natural form
        text = re.sub('[-,:]', ' ', text) # replace the character "-" "," with space
        text = re.sub("(?!<=\d)(\.)(?!\d)", '', text) # remove the character ".", except from floating numbers
        text = re.sub('[^A-Za-z0-9. ]+', '', text) # remove all punctuation, except A-Za-z0-9 
        text = re.sub(' +', ' ', text) # remove extra space
        return text

    @st.cache(allow_output_mutation=True)
    def load_data_model():
        data = pd.read_csv('mscoco_train2014_k1000_50k.csv')
        tokenizer_50k = pickle.load(open('model/tokenizer_50k.pkl', 'rb'))
        labelencoder = pickle.load(open('model/labelencoder.pkl', 'rb'))
        model = tf.keras.models.load_model('model/model_2lstm_vgg19_50k_1011_50.h5')
        return data, tokenizer_50k, labelencoder, model
    
    def final_function_1(X): 
        
        if (X[0] is None) or ((X[1] is None)):
            return " "
    
        que_clean_text = text_preprocess(X[0])
        que_vector = pad_sequences(tokenizer.texts_to_sequences([que_clean_text]), maxlen=22, padding='post') 
        
        if type(X[1]) == str:
            img = cv2.imread(X[1])
            img = cv2.resize(img,(224,224),interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(X[1])
            img = img.resize((224,224))
            img = np.array(img)

        img_vector = (img/255.0).astype(np.float32)
        predicted_Y = model.predict([que_vector,np.array([img_vector])],verbose=0)
        predicted_class = tf.argmax(predicted_Y, axis=1, output_type=tf.int32)
        predicted_ans = labelencoder.inverse_transform(predicted_class)

        return predicted_ans[0]
    
    data, tokenizer, labelencoder, model = load_data_model()
    
    option = st.sidebar.radio("Select or Upload Question & Image",('Select', 'Upload'))

    if option == 'Select':
        selected_que = st.sidebar.selectbox('Select a question',list(set(data['question'])))
        selected_img_path = st.sidebar.selectbox('Select an image',list((data[data.question == selected_que])['image_id']))
        selected_img = Image.open(selected_img_path)    
    elif option == 'Upload':
        selected_que = st.sidebar.text_input('Type a question')
        selected_img = st.sidebar.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    
    if selected_que is None:
        st.write('Type a question')
    else:
        st.write('**Question:**',selected_que)
    
    if selected_img is None:
        st.write('Upload an image')
    else:
        st.image(selected_img)
    
    if option == 'Select':
        actual_ans = list((data[(data.image_id == selected_img_path) & (data.question == selected_que)])['answer'])[0]
        st.write('**Actual Answer:**',actual_ans)
        
        predicted_ans = final_function_1([selected_que, selected_img_path])
        st.write('**Predicted Answer:**',predicted_ans)  
        
    elif option == 'Upload':
        predicted_ans = final_function_1([selected_que, selected_img])
        st.write('**Predicted Answer:**',predicted_ans)
    
    
if __name__ == "__main__":
    main()
    