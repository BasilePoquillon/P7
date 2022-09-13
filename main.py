import streamlit as st
import pickle
import pandas as pd
import lime.lime_tabular  

st.write('''
# Est-ce que le client a droit à son crédit
''')

st.sidebar.header("Les parametres d'entrée")

client_id=st.sidebar.number_input('L\'id du client')


df = pd.read_csv(r'C:\Users\Basile Poquillon\Documents\Formation\P7 Basile Poquillon\df_travail_update.csv')


def custom_metric(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = mat.ravel()
    Mesure = ((tn)-(fn + fp*10))/(len(y_pred))
    
    return Mesure        
    
loaded_model = pickle.load(open(r'C:\Users\Basile Poquillon\Documents\Formation\P7 Basile Poquillon\mypicklefile', 'rb'))

client_info = df[df['SK_ID_CURR']==client_id]
client_info = client_info.drop(['SK_ID_CURR'], axis=1)

prediction=loaded_model.predict(client_info)
if prediction == 0:
    st.subheader('Le client va probablement rembourser.')
if prediction == 1:
    st.subheader('Le client ne va probablement pas rembourser.')

    
X_train = pickle.load(open(r'C:\Users\Basile Poquillon\Documents\Formation\P7 Basile Poquillon\X_train', 'rb'))
y_train = pickle.load(open(r'C:\Users\Basile Poquillon\Documents\Formation\P7 Basile Poquillon\y_train', 'rb'))
    
    
title_text = 'Raisons de ce choix'

st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)

st.text("")

if st.button("Explain Results"):
    with st.spinner('Calculating...'):
        exp = lime.lime_tabular.LimeTabularExplainer(X_train.values, 
                                                               mode='regression',
                                                               training_labels=y_train, 
                                                               feature_names=df.columns.values, 
                                                               categorical_features=['CHAS'])
        explanation = exp.explain_instance(client_info, loaded_model.predict)
        # Display explainer HTML object
        st.components.v1.html(explanation.as_html(), height=800)