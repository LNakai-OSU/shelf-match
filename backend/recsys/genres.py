"""
The UCSD Goodreads Book Graph ships books already bucketed into ~10 clean
top-level genres (goodreads_book_genres_initial.json.gz), unlike
goodbooks-10k which only had raw user-generated shelf tags ("ya-fantasy",
"sci-fi-suspense", 300+ of them) that had to be hand-collapsed into
canonical genres (see the retired genre_tags.py). This just maps those ~10
buckets to friendlier section labels for the UI.
"""

SECTION_LABELS = {
    "children": "Children's",
    "comics, graphic": "Graphic Novels & Comics",
    "fantasy, paranormal": "Sci-Fi & Fantasy",
    "fiction": "Fiction",
    "history, historical fiction, biography": "History & Biography",
    "mystery, thriller, crime": "Mystery & Thriller",
    "non-fiction": "Nonfiction",
    "poetry": "Poetry",
    "romance": "Romance",
    "young-adult": "Young Adult",
    "": "General",
}


def section_for(primary_genre):
    return SECTION_LABELS.get(primary_genre or "", "General")
