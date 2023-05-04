from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_pymongo import PyMongo
import bcrypt
from dotenv import load_dotenv
import os
import movie_poster

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
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    if request.method == 'GET':
        # Fetch movie data and store it in a list

        # top_K_list = [1,2,3] -> 'models output' 
        # for k in top_K_list:
        #     movie_name = DBcallById()
        movie_list = ["Batman", "The Avengers", "Toy Story"] # to be replaced by csv file
        # ratings = [null, 3, 4] # display ratings

        movies_data = []
        for movie_name in movie_list:
            movie_data = {}
            movie_data["title"] = movie_name
            movie_data["poster_path"] = movie_poster.get_movie_poster_url(movie_name)
            movies_data.append(movie_data)

        # Render the home.html template and pass in the movie data
        return render_template("home.html", movies_data=movies_data)

@app.route('/logout', methods = ['POST'])
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))



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
                return redirect(url_for('home'))
            else:
                error = 'Invalid password'
                return render_template('login.html', error = error)
        else:
            error = 'Email not found'
            return render_template('login.html', error=error)
    else:
        return render_template('login.html')



@app.route('/matrix', methods=['POST'])
def create_matrix():
    data = request.get_json()
    rows = data['rows']
    cols = data['cols']
    matrix = data['data']
    # document-based format
    matrix_doc = {
        'name': 'URM',
        'matrix': matrix,
        'rows': rows,
        'cols': cols
    }
    mongo.db.matrices.insert_one(matrix_doc)
    return jsonify({'message': 'Matrix created successfully!'})

@app.route('/matrix', methods=['GET'])
def fetch_matrix():
    matrix = mongo.db.matrices.find_one({'name':'URM'})
    # print(matrix)
    return 

if __name__ == '__main__':
    app.run(debug=True)

