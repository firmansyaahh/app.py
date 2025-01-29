import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Setting untuk memilih algoritma
algorithms = ['Pilih Algoritma', 'Support Vector Machine (SVM)', 'Logistic Regression (LR)']
selected_algorithm = st.selectbox('Pilih Algoritma', algorithms)

if selected_algorithm != 'Pilih Algoritma':
    st.write(f'Anda memilih algoritma: {selected_algorithm}')
    
    df = pd.read_csv('diabetes.csv')
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if selected_algorithm == 'Support Vector Machine (SVM)':
        st.write("Implementasi Support Vector Machine (SVM)")
        st.image('header_image.png', caption='Support Vector Machine (SVM)', use_column_width=True)

        try:
            svm_model = pickle.load(open('svm_model.sav', 'rb'))
        except FileNotFoundError:
            svm_model = SVC()
            svm_model.fit(X_train, y_train)
            pickle.dump(svm_model, open('svm_model.sav', 'wb'))

        svm_pred = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        st.write(f"Akurasi model SVM: {svm_accuracy:.2f}")

        st.subheader('Prediksi Diabetes  dengan SVM')
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input('Pregnancies', min_value=0, value=0)
            glucose = st.number_input('Glucose', min_value=0, value=0)
            blood_pressure = st.number_input('BloodPressure', min_value=0, value=0)
            skin_thickness = st.number_input('SkinThickness', min_value=0, value=0)
        with col2:
            insulin = st.number_input('Insulin', min_value=0, value=0)
            bmi = st.number_input('BMI', min_value=0.0, value=0.0)
            diabetes_pedigree = st.number_input('DiabetesPedigreeFunction', min_value=0.0, value=0.0)
            age = st.number_input('Age', min_value=0, value=0)
        
        if st.button('Prediksi dengan SVM'):
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]], columns=X.columns)
            prediction = svm_model.predict(input_data)
            st.success("Hasil Prediksi: Diabetes" if prediction[0] == 1 else "Hasil Prediksi: Tidak Diabetes")

    elif selected_algorithm == 'Logistic Regression (LR)':
        st.write("Implementasi Logistic Regression (LR)")
        st.image('header_image.png', caption='Logistic Regression (LR)', use_column_width=True)
        
        lr_model = LogisticRegression (max_iter=1000)
        lr_model.fit(X_train, y_train)
        
        lr_pred = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        st.write(f"Akurasi model Logistic Regression: {lr_accuracy:.2f}")
        
        st.subheader('Confusion Matrix')
        conf_matrix = confusion_matrix(y_test, lr_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        
        st.subheader('Classification Report')
        st.text(classification_report(y_test, lr_pred))
        
        st.subheader('Prediksi Diabetes dengan Logistic Regression')
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input('Pregnancies', min_value=0, value=0)
            glucose = st.number_input('Glucose', min_value=0, value=0)
            blood_pressure = st.number_input('BloodPressure', min_value=0, value=0)
            skin_thickness = st.number_input('SkinThickness', min_value=0, value=0)
        with col2:
            insulin = st.number_input('Insulin', min_value=0, value=0)
            bmi = st.number_input('BMI', min_value=0.0, value=0.0)
            diabetes_pedigree = st.number_input('DiabetesPedigreeFunction', min_value=0.0, value=0.0)
            age = st.number_input('Age', min_value=0, value=0)
        
        if st.button('Prediksi dengan Logistic Regression'):
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]], columns=X.columns)
            prediction = lr_model.predict(input_data)
            st.success("Hasil Prediksi: Diabetes" if prediction[0] == 1 else "Hasil Prediksi: Tidak Diabetes")
