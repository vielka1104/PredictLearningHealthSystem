import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .models import Model

scaler = StandardScaler()
regressor = RandomForestClassifier(n_estimators=20, random_state=0)


def get_model(data):
    pregnancies = data.get('pregnancies', None)
    glucose = data.get('glucose', None)
    blood_pressure = data.get('blood_pressure', None)
    skin_thickness = data.get('skin_thickness', None)
    insulin = data.get('insulin', None)
    bmi = data.get('bmi', None)
    diabetes_pedigree_function = data.get('diabetes_pedigree_function', None)
    age = data.get('age', None)
    model = Model(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function,
                  age)
    return model


def classifier():
    url = 'https://raw.githubusercontent.com/MarioTataje/lhs-dataset/main/ckd.csvp'
    dataset = pd.read_csv(url)
    features = dataset.iloc[:, 0:8].values
    labels = dataset.iloc[:, 8].values
    train_features, test_attributes, train_labels, test_labels = train_test_split(features, labels, test_size=0.9,
                                                                                  random_state=0)
    train_features = scaler.fit_transform(train_features)
    regressor.fit(train_features, train_labels)


def predict(model):
    test_attributes = get_test(model)
    test_attributes = scaler.transform(test_attributes)
    prediction = regressor.predict(test_attributes)[0]
    possibilities = regressor.predict_proba(test_attributes)[0]
    return get_response(prediction, possibilities[1], possibilities[0])


def get_test(model):
    test = [[model.pregnancies, model.glucose, model.blood_pressure, model.skin_thickness, model.insulin, model.bmi,
            model.diabetes_pedigree_function, model.age]]
    return test


def get_response(outcome, yes_possibility, no_possibility):
    return {'outcome': int(outcome), 'yes': yes_possibility, 'no': no_possibility}
