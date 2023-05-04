import requests
from dotenv import load_dotenv
import os

load_dotenv()

base_url = "http://www.omdbapi.com/"
api_key = os.getenv("OMDB_API_KEY")

def get_movie_poster_url(movie_name):
    response = requests.get(base_url, params={"apikey": api_key, "t": movie_name})
    data = response.json()
    poster_url = data["Poster"]
    return poster_url
