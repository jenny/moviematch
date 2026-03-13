import os
import requests
import json
import time # todo add sleep to tmdb calls
import math

from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv() # Reload the .env file
CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")
TMDB_KEY = os.getenv("TMDB_READ_ACCESS_TOKEN")
if not TMDB_KEY:
    raise ValueError("TMDB_READ_ACCESS_TOKEN is not set. Check your .env file.")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_HEADERS = headers={
    "accept": "application/json",
    "Authorization": "Bearer " + TMDB_KEY
}

# already initialized data for top 100 movies, don't need to do again
def initialize_app_data(n):
    top_movie_ids = initialize_index(n)
    for movie in top_movie_ids:
        initialize_movie_metadata(movie)
    # todo: remove the for loop from the function below
    # and use the for loop above
    initialize_movie_richtext()


def initialize_index(n):
    pages = math.ceil(n/20)
    ids = []
    index = {"results": []}


    with open("data/index.json", "w") as file:
        for page in range (1,pages+1):
            # print("page " + str(page))
            response = requests.get(
                TMDB_BASE_URL + "/movie/top_rated",
                params = {
                    "page": str(page)
                },
                headers = TMDB_HEADERS
            )
            response.raise_for_status()
            for movie in response.json()["results"]:
                if len(ids) < n: 
                    ids += [movie["id"]]
                    # print(str(len(ids)) + " " + str(movie["id"]) + " " + movie["title"])
    
                    # write an index of all fetched ids to file
                    index["results"].append({"id": movie["id"], "title": movie["title"]})
                    
        json.dump(index, file, indent=2)
        print("Dumped indexes for " + str(len(index["results"])) + " movies to data/index.json.")
    return ids


def initialize_movie_metadata(movie_id):
    response = requests.get(
        TMDB_BASE_URL + "/movie/"+str(movie_id)+"?append_to_response=keywords,credits",
        headers = TMDB_HEADERS
    )
    response.raise_for_status()
    movie_json = response.json()
    
    # filter cast here
    movie_json = filter_movie_cast(movie_json)

    # filter crew here
    movie_json = filter_and_sort_movie_crew(movie_json)
    
    with open("data/" + str(movie_id) + ".json", "w") as file:
        json.dump(movie_json, file, indent=2)
        print("Dumped filtered movie metadata for " + str(movie_json["title"]) + " to " + str(movie_id) + ".json")

def filter_movie_cast(movie_json):
    cast = movie_json["credits"]["cast"]
    cast = cast[0:30]  # only keep top 30 billed actors
    movie_json["credits"]["cast"] = cast
    return movie_json


def filter_and_sort_movie_crew(movie_json):
    crew = movie_json["credits"]["crew"]
    filtered_crew = []
    for member in crew:
        if (member["job"] == "Director" or
            member["job"] == "Executive Producer" or
            member["job"] == "Producer"):
            filtered_crew.append(member)
    sorted_crew = sorted(filtered_crew, key = lambda c: c["job"])
    movie_json["credits"]["crew"] = sorted_crew
    return movie_json

# Compiles a 'richtext' string for each movie: 
# title + plot summary + genre names + top 5 keywords into a single string
# and inserts into each movies's json file
def initialize_movie_richtext():
    with open("data/index.json", "r") as index:
        
        movies = json.load(index)
        
        for movie in movies["results"]:
            file_name = "data/" + str(movie["id"]) + ".json"
            movie_json = {}
            rich_text = ""

            with open(file_name, "r") as file:
                movie_json = json.load(file)



                rich_text = "Plot: " + movie_json["overview"] + "\n\n"
                rich_text += "Themes and Keywords: "
                rich_text += ', '.join(k["name"] for k in movie_json["keywords"].get("keywords", []))
                rich_text += "\n\n"
                #rich_text += "Genres: "
                #rich_text += ', '.join(g["name"] for g in movie_json.get("genres", []))
                #rich_text += "\n\n"
                #for crew in movie_json["credits"]["crew"]:
                #    rich_text += crew["job"] + ": " + crew["name"] + "\n\n"
                #rich_text += "Top Cast: "
                #rich_text += ', '.join(c["name"] for c in movie_json["credits"].get("cast", []))
                #rich_text += "\n\n"
                #rich_text += "Title: " + movie_json["title"] + " (" + movie_json["release_date"][:4] + ")\n\n"
                #todo: add belongs to collection

                movie_json["richtext"] = rich_text
            
            with open(file_name, "w") as file:
                json.dump(movie_json, file, indent=2)
                print("Dumped rich text for " + str(movie_json["title"]) + " to " + str(movie_json["id"]) + ".json")


# richtext word count
def debug_wordcount(field):
    print("Wordcounts for " + field)
    with open("data/index.json", "r") as index:
            
            movies = json.load(index)
            
            for movie in movies["results"]:
                file_name = "data/" + str(movie["id"]) + ".json"
                with open(file_name, "r") as file:
                    movie_json = json.load(file)
                    word_count = len(movie_json[field].split())
                    print(str(word_count) + ": " + str(movie["title"]) + " (" + str(movie["id"]) + ")")

if __name__ == "__main__":
    initialize_movie_richtext()

