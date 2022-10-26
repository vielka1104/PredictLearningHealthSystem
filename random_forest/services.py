import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from .models import Model

regressor = RandomForestClassifier(n_estimators=20, random_state=0)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')


def get_model(data):
    age = data.get('age', None)
    blood_pressure = data.get('blood_pressure', None)
    specific_gravity = data.get('specific_gravity', None)
    albumin = data.get('albumin', None)
    sugar = data.get('sugar', None)
    red_blood_cells = data.get('red_blood_cells', None)
    pus_cell = data.get('pus_cell', None)
    pus_cell_clumps = data.get('pus_cell_clumps', None)
    bacteria = data.get('bacteria', None)
    blood_glucose_random = data.get('blood_glucose_random', None)
    blood_urea = data.get('blood_urea', None)
    serum_creatinine = data.get('serum_creatinine', None)
    sodium = data.get('sodium', None)
    potassium = data.get('potassium', None)
    hemoglobin = data.get('hemoglobin', None)
    packed_cell_volume = data.get('packed_cell_volume', None)
    white_blood_cell_count = data.get('white_blood_cell_count', None)
    red_blood_cell_count = data.get('red_blood_cell_count', None)
    hypertension = data.get('hypertension', None)
    diabetes_mellitus = data.get('diabetes_mellitus', None)
    coronary_artery_disease = data.get('coronary_artery_disease', None)
    appetite = data.get('appetite', None)
    pedal_edema = data.get('pedal_edema', None)
    anemia = data.get('anemia', None)
    model = Model(age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell,
                  pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium, potassium,
                  hemoglobin, packed_cell_volume, white_blood_cell_count, red_blood_cell_count, hypertension,
                  diabetes_mellitus, coronary_artery_disease, appetite, pedal_edema, anemia)
    return model


def classifier():
    url = 'https://raw.githubusercontent.com/MarioTataje/lhs-dataset/main/ckd.csv'
    dataset = pd.read_csv(url)
    features = dataset.iloc[:, 0:24].values
    labels = dataset.iloc[:, 24].values
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.1,
                                                                                random_state=0)
    global imp
    imp = imp.fit(train_features)
    train_features = imp.transform(train_features)
    regressor.fit(train_features, train_labels)


def predict(model):
    test_feature = get_test(model)
    test_feature = imp.transform(test_feature)
    prediction = regressor.predict(test_feature)[0]
    possibilities = regressor.predict_proba(test_feature)[0]
    return get_response(prediction, possibilities[1], possibilities[0])


def get_test(model):
    test = [[model.age, model.blood_pressure, model.specific_gravity, model.albumin, model.sugar,
             model.red_blood_cells, model.pus_cell, model.pus_cell_clumps, model.bacteria, model.blood_glucose_random,
             model.blood_urea, model.serum_creatinine, model.sodium, model.potassium, model.hemoglobin,
             model.packed_cell_volume, model.white_blood_cell_count, model.red_blood_cell_count, model.hypertension,
             model.diabetes_mellitus, model.coronary_artery_disease, model.appetite, model.pedal_edema, model.anemia]]
    return test


def get_response(outcome, yes_possibility, no_possibility):
    return {'outcome': int(outcome), 'yes': yes_possibility, 'no': no_possibility}
