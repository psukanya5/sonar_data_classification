import numpy as np
import streamlit as st
import pickle
import re #regular expression
with open('model.pkl','rb') as f:
    model = pickle.load(f)
st.title('SONAR DATA CLASSIFICATION')
#list comprehension
collect_numbers = lambda x : [float(i) for i in re.split(",", x)]
#getting input
input_data = st.text_input("Enter the input data separated by commas")
num_list = []
num_list = collect_numbers(input_data)
#changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(num_list)

#reshaping numpy array for predicting one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#making a predictive system
if st.button("Classify") :
    predictions = model.predict(input_data_reshaped)
    if predictions == 'R':
        st.write("The object is a Rock")
    else:
        st.write("The object is a Mine")




