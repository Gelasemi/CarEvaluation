# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fonction pour charger les données avec le cache Streamlit
@st.cache_data
def load_data():
    file_path = r"C:\Users\hp\Desktop\PROJET MANJAKA\car.data"
    data = pd.read_csv(file_path, header=None)
    data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    return data

# Fonction pour prétraiter les données
def preprocess_data(data):
    le = LabelEncoder()
    encoded_data = data.apply(le.fit_transform)
    X = encoded_data.iloc[:, :-1]
    y = encoded_data.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Interface utilisateur
st.title("Application de Prévision - Car Evaluation 🚗")
st.write("""
Cette application prédit la classe d'un véhicule selon 6 caractéristiques techniques.
Les classes possibles sont : inacceptable, acceptable, bien, très bien.
""")

# Chargement des données
data = load_data()
st.write("### Aperçu des Données (5 premières lignes)", data.head())

# Entraînement du modèle
X_train, X_test, y_train, y_test = preprocess_data(data)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Performance du modèle :** {accuracy * 100:.2f}% de précision")

# Formulaire de prédiction
with st.form("prediction_form"):
    st.write("### Entrez les caractéristiques du véhicule")
    
    col1, col2 = st.columns(2)
    with col1:
        buying = st.selectbox("Coût d'achat", ['vhigh', 'high', 'med', 'low'])
        maint = st.selectbox("Coût d'entretien", ['vhigh', 'high', 'med', 'low'])
        doors = st.selectbox("Nombre de portes", ['2', '3', '4', '5more'])
    
    with col2:
        persons = st.selectbox("Passagers", ['2', '4', 'more'])
        lug_boot = st.selectbox("Taille du coffre", ['small', 'med', 'big'])
        safety = st.selectbox("Sécurité", ['low', 'med', 'high'])
    
    submitted = st.form_submit_button("Prédire")
    
    if submitted:
        # Encodage
        input_data = pd.DataFrame([[buying, maint, doors, persons, lug_boot, safety]],
                                columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
        
        le_dict = {}
        for col in input_data.columns:
            le = LabelEncoder()
            le.fit(data[col])
            input_data[col] = le.transform(input_data[col])
            le_dict[col] = le
        
        # Prédiction
        prediction = model.predict(input_data)
        classes = ["inacceptable", "acceptable", "bien", "très bien"]
        
        # Résultat
        st.success(f"Résultat de la prédiction : **{classes[prediction[0]]}**")
        st.write("Détails d'encodage :")
        st.json({col: list(le_dict[col].classes_) for col in input_data.columns})