import numpy as np
import pickle
import streamlit as st

load_model=pickle.load(open('C:/webappmachinelearning/train_model.sav','rb'))

# creating a function for prediction

def diabetes_prediction(input):
    
    input_as_numpy=np.asarray(input)
    input_reshape=input_as_numpy.reshape(1,-1)
    prediction=load_model.predict(input_reshape)
    print(prediction)
    if (prediction[0] == 0):
      return" Patient has no diabetes"
    else:
      return" Patient has diabetes"
      
# create main function 
def main():
    st.set_page_config(
        page_title="diabetes prediction",
        layout="centered")
    
    st.title("Diabetes Prediction System")
    st.subheader("Go and Check for diabetes")
    
    
    Pregnancies = st.text_input("Enter no. of Pregnancies")
    Glucose = st.text_input("Number of Glucose level")
    BloodPressure = st.text_input('BP level')
    SkinThickness = st.text_input(" Skin thickness")
    Insulin = st.text_input('insulin')
    BMI = st.text_input(" bmi ")
    DiabetesPedigreeFunction = st.text_input("diabets pedigree function")
    Age = st.text_input("age")
    
    # code for prediction
    
    diagnosis = ''
    
    if st.button("Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness,
                                         Insulin, BMI,
                                         DiabetesPedigreeFunction,Age ])
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    