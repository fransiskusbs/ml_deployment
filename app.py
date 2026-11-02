from fastapi import FastAPI
import pickle
import pandas as pd
import sys
print('Python', sys.executable)

app = FastAPI(title='Home Credit Prediction')

def load_model():
    with open('model_rf_v2.sav', 'rb') as file:
        return pickle.load(file)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/get_predict')
def get_prediction(request:list[dict]):
    data = pd.DataFrame(request)
    model = load_model()
    prediction_prob = model.predict_proba(data)[:,1].tolist()
    prediction_class = [1 if 1>= 0.5 else 0 for i in prediction_prob]
    return{'class': prediction_class, 'prob': prediction_prob}
