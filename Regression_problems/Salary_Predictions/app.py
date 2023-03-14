import pickle
from flask import Flask, request, render_template
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('saved_model\\best_rfg.pkl', 'rb'))
sc_rating = pickle.load(open('saved_model\\sc_rating.pkl', 'rb'))
sc_founded = pickle.load(open('saved_model\\sc_founded.pkl', 'rb'))

@app.route('/')
# @cross_origin()
def homepage():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
# @cross_origin()
def predict_salary():
    if request.method == 'POST':
        final_input = []


        #     # company rating
        #     rating = sc_rating.fit_transform(np.array(rating).reshape(1,-1))
        #     final_input = np.concatenate((final_input, rating[0]))
        rating = request.form['Company_Rating']
        rating = sc_rating.transform(np.array(rating).reshape(1, -1))
        final_input.append(rating[0][0])

        #     # company founded
        #     founded = sc_founded.fit_transform(np.array(founded).reshape(1, -1))
        #     final_input = np.concatenate((final_input, founded[0]))
        founded = request.form['Company_Founded_Year']
        founded = sc_founded.transform(np.array(founded).reshape(1, -1))
        final_input.append(founded[0][0])

        # seniority
        seniority = request.form['Seniority']

        if seniority == 'other':
            final_input = np.concatenate((final_input, [0]))
        elif seniority == 'jr':
            final_input = np.concatenate((final_input, [1]))
        elif seniority == 'sr':
            final_input = np.concatenate((final_input, [2]))

        # job skills
        skills = request.form.getlist('skills')

        temp = list(map(int, np.zeros(shape=(1, 4))[0]))
        if 'excel' in skills:
            temp[0] = 1
        if 'python' in skills:
            temp[1] = 1
        if 'tableau' in skills:
            temp[2] = 1
        if 'sql' in skills:
            temp[3] = 1
        final_input = np.concatenate((final_input, temp))

        # headquarters
        job_in_headquarter = request.form['Job_in_Headquarter']

        if job_in_headquarter == 'yes':
            final_input = np.concatenate((final_input, [1]))
        elif job_in_headquarter == 'no':
            final_input = np.concatenate((final_input, [0]))

        # revenue
        revenue = request.form['Company_Revenue']

        if revenue == 'Unknown / Non-Applicable':
            final_input = np.concatenate((final_input, [0]))
        elif revenue == 'Less than $1 million (USD)':
            final_input = np.concatenate((final_input, [1]))
        elif revenue == '$1 to $5 million (USD)':
            final_input = np.concatenate((final_input, [2]))
        elif revenue == '$5 to $10 million (USD)':
            final_input = np.concatenate((final_input, [3]))
        elif revenue == '$10 to $25 million (USD)':
            final_input = np.concatenate((final_input, [4]))
        elif revenue == '$25 to $50 million (USD)':
            final_input = np.concatenate((final_input, [5]))
        elif revenue == '$50 to $100 million (USD)':
            final_input = np.concatenate((final_input, [6]))
        elif revenue == '$100 to $500 million (USD)':
            final_input = np.concatenate((final_input, [7]))
        elif revenue == '$500 million to $1 billion (USD)':
            final_input = np.concatenate((final_input, [8]))
        elif revenue == '$1 to $2 billion (USD)':
            final_input = np.concatenate((final_input, [9]))
        elif revenue == '$2 to $5 billion (USD)':
            final_input = np.concatenate((final_input, [10]))
        elif revenue == '$5 to $10 billion (USD)':
            final_input = np.concatenate((final_input, [11]))

        # competitors
        Competitors = int(request.form['Number_of_Competitors'])

        final_input = np.concatenate((final_input, [Competitors]))

        # job Title
        Job_title = request.form['Job_Title']

        job_title_columns = ['job_title_data analyst', 'job_title_data scientist', 'job_title_director',
                             'job_title_manager', 'job_title_ml engineer']
        temp = list(map(int, np.zeros(shape=(1, len(job_title_columns)))[0]))
        for index in range(0, len(job_title_columns)):
            if job_title_columns[index] == 'job_title_' + Job_title:
                temp[index] = 1
                break
        final_input = np.concatenate((final_input, temp))

        # Type of ownership
        Ownership = request.form['Type_of_Ownership']

        ownership_columns = ['type_of_ownership_College / University', 'type_of_ownership_Government',
                             'type_of_ownership_Hospital', 'type_of_ownership_Nonprofit Organization',
                             'type_of_ownership_Private', 'type_of_ownership_Subsidiary or Business Segment']
        temp = list(map(int, np.zeros(shape=(1, len(ownership_columns)))[0]))
        for index in range(0, len(ownership_columns)):
            if ownership_columns[index] == 'type_of_ownership_' + Ownership:
                temp[index] = 1
                break
        final_input = np.concatenate((final_input, temp))

        # Sector
        Sector = request.form['Type_of_Sector']

        sector_columns = ['sector_Aerospace & Defense', 'sector_Biotech & Pharmaceuticals', 'sector_Business Services',
                          'sector_Finance', 'sector_Health Care', 'sector_Information Technology', 'sector_Insurance',
                          'sector_Manufacturing']
        temp = list(map(int, np.zeros(shape=(1, len(sector_columns)))[0]))
        for index in range(0, len(sector_columns)):
            if sector_columns[index] == 'sector_' + Sector:
                temp[index] = 1
                break
        final_input = np.concatenate((final_input, temp))
        print(final_input)

        prediction = regmodel.predict(np.array(final_input).reshape(1, -1))

        #      round it to 5 decimals only
        salary = round(prediction[0], 2)

        return render_template('prediction.html', prediction=salary)
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
