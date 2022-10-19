import sklearn
import streamlit as st
import pickle
from PIL import Image

st.header("❀ Iris Species Prediction !")
st.sidebar.subheader("Using KNN")

sepalLength = st.sidebar.slider('Sepal Length in cm ',1.0,10.0,5.1)
sepalWidth = st.sidebar.slider('Sepal Width in cm ',1.0,5.0,3.5)
petalLength = st.sidebar.slider('petal Length in cm ',1.0,10.0,1.4)
petalwidth = st.sidebar.slider('petal Width in cm ',0.0,5.0,0.2)

if st.sidebar.button(' Get Prediction '):
    load_model = pickle.load(open('knn.pkl','rb'))
    res = load_model.predict([[sepalLength,sepalWidth,petalLength,petalwidth]])
    
    if res == 'Iris-setosa':
        st.success(f"Iris-Species : {res} ❀")
        image = Image.open('images/Iris-setosa.jpg')
        st.image(image, caption='iris-setosa')
        
    elif res == 'Iris-versicolor':
        st.success(f"Iris-Species : {res} ❀")
        image = Image.open('images/Iris-versicolor.jpg')
        st.image(image, caption='iris-versicolor')
        
    else :
        st.success(f"Iris-Species : {res} ❀")
        image = Image.open('images/Iris-virginica.jpg')
        st.image(image, caption='iris-virginica')
        





