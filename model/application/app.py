from retrain_model import retrainModel
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, make_response
from flask_pymongo import PyMongo
import bcrypt
from dotenv import load_dotenv
import os
import json
import requests
# from model import run_example
# import sys
# sys.path.append('/Users/varunjain/Desktop/Jumpstart-BTP/')
# path="/Users/varunjain/Desktop/Jumpstart-BTP/model"
from gettopk import getTopK
# export PYTHONPATH="$PYTHONPATH:/Users/varunjain/Desktop/Jumpstart-BTP/model"
# from gettopk import getTopK

load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')

password = os.getenv("PASSWORD")
app.secret_key = os.getenv("APP_SECRET_KEY")

app.config['MONGO_URI'] = 'mongodb+srv://alaammfarhan:'+password+'@cluster0.xlweciy.mongodb.net/test'

mongo = PyMongo(app)

@app.route("/")
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('home'))

@app.route("/home", methods=['GET', 'POST'])
def home():
    if 'logged_in' not in session or session['logged_in'] == False:
        return redirect(url_for('login'))
    if request.method == 'GET':
        print_session_detail()
        userId = int(session['userId'])
        movie_ids=getTopK(userId);
        # movie_ids=[1,2,3,4]
        # print(movie_ids)
        votes_cookie = request.cookies.get('votes')
        if votes_cookie:
            movie_ids = []
            vote_values = []
            votes = json.loads(votes_cookie)
            for movie_id, vote_value in votes.items():
                movie_ids.append(movie_id)
                vote_values.append(vote_value)
                print(f"Movie ID {movie_id}: Vote value {vote_value}")
            result=retrainModel(userId,movie_ids,vote_values);
            print(result);
        # print(result);
    
        movies_data = []
        for movie_id in movie_ids:
            movie_id_str=str(movie_id);
            movie = mongo.db.csvs.find_one({'movieId':movie_id_str})
            movieName = str(movie["title"])
            movie_name_without_year = movieName[:-6]
            movie_data = {}
            movie_data["title"] = movie_name_without_year
            movie_data["poster_path"] = get_movie_poster_url(movie_name_without_year)
            movie_data["movie_id"] = movie_id
            movies_data.append(movie_data)

        # Render the home.html template and pass in the movie data
        return render_template("home.html", movies_data=movies_data)



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        # Hash the password
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())

        # Check if user already exists
        user = mongo.db.users.find_one({'email': email})
        if user:
            return 'Email already exists'

        # Add the user to the database
        mongo.db.users.insert_one({
            'name': name,
            'email': email,
            'password': hashed_password
        })

        return redirect(url_for('login'))
    else:
        return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        user = mongo.db.users.find_one({'email': email})
        if user:
            # Verify password
            if bcrypt.checkpw(password, user['password']):
                session['logged_in'] = True
                session['email'] = email
                session['userId']=str(user["userId"])
                print("Login")
                print(session['userId']);
                return redirect(url_for('home'))
            else:
                error = 'Invalid password'
                return render_template('login.html', error = error)
        else:
            error = 'Email not found'
            return render_template('login.html', error=error)
    else:
        return render_template('login.html')

@app.route('/vote', methods=['POST'])
def vote():
    data = request.get_json()
    movie_id = int(data['movie_id'])
    vote_value = int(data['vote_value'])
    # do something with the vote data, like updating a database

    # Get the votes dictionary from the cookie or create a new one if it doesn't exist
    votes_cookie = request.cookies.get('votes')
    if votes_cookie:
        votes = json.loads(votes_cookie)
    else:
        votes = {}

    # Update the votes dictionary with the new vote
    votes[movie_id] = vote_value
    # print(votes[movie_id])
    # print(len(votes))
    

    # Save the updated votes dictionary to a cookie and return a response
    response = make_response('OK')
    response.set_cookie('votes', value=json.dumps(votes))
    
    print_session_detail()
    return response


@app.route('/logout', methods = ['POST'])
def logout():
    print("Logout")
    print_session_detail()
    session.pop('logged_in', None)
    session.pop('userId', None)
    session.pop('email',None)
    # session.pop('votes',None)
    response = make_response(redirect(url_for('login')))
    response.delete_cookie('votes')
    return response

# @app.route('/matrix', methods=['POST'])
# def create_matrix():
#     data = request.get_json()
#     rows = data['rows']
#     cols = data['cols']
#     matrix = data['data']
#     # document-based format
#     matrix_doc = {
#         'name': 'URM',
#         'matrix': matrix,
#         'rows': rows,
#         'cols': cols
#     }
#     mongo.db.matrices.insert_one(matrix_doc)
#     return jsonify({'message': 'Matrix created successfully!'})

# @app.route('/matrix', methods=['GET'])
# def fetch_matrix():
#     matrix = mongo.db.matrices.find_one({'name':'URM'})
#     # print(matrix)
#     return 

if __name__ == '__main__':
    app.run(debug=True)



base_url = "http://www.omdbapi.com/"
api_key = os.getenv("OMDB_API_KEY")

def get_movie_poster_url(movie_name):
    response = requests.get(base_url, params={"apikey": api_key, "t": movie_name})
    data = response.json()
    poster_url = data["Poster"]
    return poster_url


def print_session_detail():
    if('userId' in session): 
        print(session['userId'])
    if('email' in session): 
        print(session['email'])
    votes_cookie = request.cookies.get('votes')
    if votes_cookie:
        votes = json.loads(votes_cookie)
        for movie_id, vote_value in votes.items():
            print(f"Movie ID {movie_id}: Vote value {vote_value}")
