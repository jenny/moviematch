import os
import requests
import json
import time
import math

from dotenv import load_dotenv
load_dotenv() # Reload the .env file

from init_richtext import initialize_movie_richtext

TMDB_KEY = os.getenv("TMDB_READ_ACCESS_TOKEN")
if not TMDB_KEY:
    raise ValueError("TMDB_READ_ACCESS_TOKEN is not set. Check your .env file.")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_HEADERS = {
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


    os.makedirs("data", exist_ok=True)
    with open("data/index.json", "w") as file:
        for page in range (1,pages+1):
            response = requests.get(
                TMDB_BASE_URL + "/discover/movie",
                params = {
                    "sort_by": "vote_average.desc",
                    "vote_count.gte": "200",
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
    time.sleep(0.25)

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


if __name__ == "__main__":
    n = int(input("How many movies to initialize? "))
    initialize_app_data(n)

