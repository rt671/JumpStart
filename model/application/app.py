from retrain_model import retrainModel
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, make_response
from flask_pymongo import PyMongo
import bcrypt
from dotenv import load_dotenv
import os
import json
import requests
import re
import numpy as np
from gettopk import getTopK

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
        session['refreshed'] = True
        print_session_detail()
        userId = int(session['userId'])

        movie_ids=retrain_model()
        print(movie_ids)
        movies_data = []
        for movie_id in movie_ids:
            movie_id_str=str(movie_id)
            # print(movie_id)
            movie = mongo.db.movies.find_one({'movieId':movie_id_str})
            movieName = str(movie["title"])
            movie_name_without_year = movieName[:-6]
            print(movie_name_without_year)
            movie_data = {}
            movie_data["title"] = movie_name_without_year
            api_data = get_movie_data(movie_name_without_year)
            movie_data["actors"] = api_data[0]
            movie_data["directors"] = api_data[1]
            movie_data["poster_path"] = api_data[2]
            movie_data["movie_id"] = movie_id
            movie_data["genres"] = movie["genres"].split('|')
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
        user_meta = mongo.db.usermetadata.find_one({'metadata_id':1})
        print(user_meta)


        mongo.db.users.insert_one({
            'name': name,
            'email': email,
            'password': hashed_password,
            'userId':user_meta['new_user_id']
        })

        usermetadata = mongo.db.usermetadata

        # Find the document you want to update based on a filter
        filter_query = {'metadata_id': 1}

        # Save the updated document back to the collection
        update_query = {"$set": {"new_user_id": user_meta['new_user_id'] + 1}}

        update_result = usermetadata.update_one(filter_query, update_query)

        session['logged_in'] = True
        session['new_user_registered'] = True
        session['email'] = email
        session['userId']=str(user_meta['new_user_id'])

        return redirect(url_for('home'))
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
    print(data['movie_id'])
    print (type((data['movie_id'])))
    movie_id = int(data['movie_id'])
    vote_value = int(data['vote_value'])
    # do something with the vote data, like updating a database

    # Get the votes dictionary from the cookie or create a new one if it doesn't exist
    votes_cookie = request.cookies.get('votes')
    if session['refreshed']==True:
          session['refreshed']=False;
          votes = {}
    elif votes_cookie:
        votes = json.loads(votes_cookie)
    else:
        session['refreshed']=False;
        votes = {}

    # Update the votes dictionary with the new vote
    votes[movie_id] = vote_value
    
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
    response = make_response(redirect(url_for('login')))

    response.delete_cookie('votes')
    return response

if __name__ == '__main__':
    app.run(debug=True)


base_url = "http://www.omdbapi.com/"
api_key = os.getenv("OMDB_API_KEY")

def get_movie_poster_url(movie_name):
    response = requests.get(base_url, params={"apikey": api_key, "t": movie_name})
    data = response.json()
    # print(data)
    if(data["Response"] == "False" or "Poster" not in data or data["Poster"] == "N/A"):
        return "https://upload.wikimedia.org/wikipedia/commons/6/64/Poster_not_available.jpg"
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

def retrain_model():
    print("in retrain")
    userId = int(session['userId'])
    votes_cookie = request.cookies.get('votes')
    print(votes_cookie)
    
    if votes_cookie:
        movie_ids = []
        vote_values = []
        votes = json.loads(votes_cookie)
        print(votes)
        for movie_id, vote_value in votes.items():
            movie_ids.append(movie_id)
            vote_values.append(vote_value)
            print(f"Movie ID {movie_id}: Vote value {vote_value}")
        result=retrainModel(userId,movie_ids,vote_values);
        # print(result);
        return result
    else:
        result=getTopK(userId)
        return result

def correct_movie_name(movie_name):
    
    movie_name_array = movie_name.split(',')
    if(len(movie_name_array)==1):
        return movie_name
    # print(movie_name_array)
    if(movie_name_array[1] == " The" or movie_name_array[1] == " A" or movie_name_array[1] == " An"):
        return (movie_name_array[1] + ' ' + movie_name_array[0])
    else: 
        return movie_name

def get_movie_data(movie_name):
    response = requests.get(base_url, params={"apikey": api_key, "t": movie_name})
    data = response.json()
    # print(data)
    actors = []
    if(data["Response"] == "False" or "Actors" not in data or data["Actors"] == "N/A"):
        actors = ["N/A"]
    else:
        actors_str = data["Actors"]
        actors = actors_str.split(',')
        actors = actors[:2]

    director = ''
    if(data["Response"] == "False" or "Director" not in data or data["Director"] == "N/A"):
        director = "N/A"
    else:
        director = data["Director"]

    poster_url = ''
    if(data["Response"] == "False" or "Poster" not in data or data["Poster"] == "N/A"):
        poster_url = "https://upload.wikimedia.org/wikipedia/commons/6/64/Poster_not_available.jpg"
    else:
        poster_url = data["Poster"]

    return [actors, director, poster_url]
