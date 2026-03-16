import os
import json
import glob

from config import DATA_DIR, RICHTEXT_CAST_LIMIT


def build_richtext(movie_json: dict) -> str:
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
    rich_text += ', '.join(c["name"] for c in movie_json["credits"].get("cast", [])[:RICHTEXT_CAST_LIMIT])
    rich_text += "\n\n"
    release_year = movie_json.get("release_date", "")[:4] or "Unknown"
    rich_text += "Title: " + movie_json["title"] + " (" + release_year + ")\n\n"
    return rich_text


def compile_all_richtexts() -> None:
    index_path = os.path.join(DATA_DIR, "index.json")
    with open(index_path, "r") as f:
        movies = json.load(f)
    for movie in movies["results"]:
        file_path = os.path.join(DATA_DIR, f"{movie['id']}.json")
        if not os.path.exists(file_path):
            print(f"Warning: skipping richtext for {movie['title']} ({movie['id']}) — file not found")
            continue
        with open(file_path, "r") as f:
            movie_json = json.load(f)
        movie_json["richtext"] = build_richtext(movie_json)
        with open(file_path, "w") as f:
            json.dump(movie_json, f, indent=2)
        print(f"Built richtext for {movie_json['title']} → {movie_json['id']}.json")


def debug_wordcount(field: str) -> None:
    print(f"Wordcounts for {field}")
    index_path = os.path.join(DATA_DIR, "index.json")
    with open(index_path, "r") as f:
        movies = json.load(f)
    for movie in movies["results"]:
        file_path = os.path.join(DATA_DIR, f"{movie['id']}.json")
        with open(file_path, "r") as f:
            movie_json = json.load(f)
        word_count = len(movie_json[field].split())
        print(f"{word_count}: {movie['title']} ({movie['id']})")


if __name__ == "__main__":
    compile_all_richtexts()
