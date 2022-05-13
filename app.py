from flask import Flask , render_template,request
import pickle

loaded_model = pickle.load(open("artifacts\model.pkl","rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/predict", methods = ["GET","POST"] )
def predict():
    Pregnancies_var = int(request.form.get("Pregnancies"))
    Glucose_var = int(request.form.get("Glucose"))
    BloodPressure_var = int(request.form.get("BloodPressure"))
    SkinThickness_var = int(request.form.get("SkinThickness"))
    Insulin_var = int(request.form.get("Insulin"))
    BMI_var = float(request.form.get("BMI"))
    PDiabetesPedigreeFunction_var = float(request.form.get("DiabetesPedigreeFunction"))
    Age_var = int(request.form.get("Age"))

    # print(Pregnancies_var,Glucose_var,BloodPressure_var,SkinThickness_var,
    #       Insulin_var,BMI_var,PDiabetesPedigreeFunction_var,Age_var)

    result = loaded_model.predict([[Pregnancies_var,Glucose_var,BloodPressure_var,
                   SkinThickness_var,Insulin_var,BMI_var,PDiabetesPedigreeFunction_var,Age_var]])
    # print(result[0])

    if result[0] == 1 :
        final_result = "Patient is Diabetes Positive"
    else :
        final_result = "Patient is Diabetes Nagative"
    
    return render_template("index.html",prediction=final_result)



if __name__ == "__main__" :
    app.run(debug=True,host="0.0.0.0",port=8080)