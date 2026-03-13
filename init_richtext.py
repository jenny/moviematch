import json

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
                rich_text += "Genres: "
                rich_text += ', '.join(g["name"] for g in movie_json.get("genres", []))
                rich_text += "\n\n"
                for crew in movie_json["credits"]["crew"]:
                    if crew["job"] == "Director":
                        rich_text += "Director: " + crew["name"] + "\n\n"
                rich_text += "Top Cast: "
                rich_text += ', '.join(c["name"] for c in movie_json["credits"].get("cast", [])[:5])
                rich_text += "\n\n"
                rich_text += "Title: " + movie_json["title"] + " (" + movie_json["release_date"][:4] + ")\n\n"
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
