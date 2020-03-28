"""
Gives the list of news sources that are available for each language.
"""
from sabelotodo.text_finder.extractors.news import ElConfidencial

SOURCES = {
    "es": [ElConfidencial]
}
