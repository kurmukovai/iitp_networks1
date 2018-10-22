import json
import requests

import pandas as pd
import numpy as np


#   per_year: bool,
#           if True, coauthors
#           will contain coathors for each year,
#           if False, see "year" parameter
#     year: int, 
#           if "per_year" is False return coauthors for given year,
#           if None, return coauthors for all years
# Я передумал подавать в качестве аргументов start_year, end_year
# потому что могут возникнуть и другие столбцы вроде тематики или пола
# проще в саму функцию передавать уже обработанный дата фрейм (нужные года, и пр.)

def get_coauthors(df_data):
    """
    input
    
    df_data: pandas Data Frame,
             with columns: title, year, author_id_new
                
    return
    
    adjacency_matrix: numpy ndarray, 
                      adjacency coauthorship matrix
    authors: authors name list (indices are the same as in adjacency matrix)
    n_papers: numpy ndarray,
              total number of papers for each author
    
    EXAMPLE:
    # return coauthorship matrix for 2013-2015 years 
    
    itas2013_2015 = itas_data.loc[(itas_data.year >= 2013) & (itas_data.year <= 2015)].copy()
    adj, authors, n_papers = get_coauthors(itas2013_2015) 
    
    """

    title_author_counts = df_data.groupby(by=['title', 'author_id_new']).count()

    titles, authors = title_author_counts.index.levels
    title_index, author_index = title_author_counts.index.labels

    incidence_matrix = pd.crosstab(author_index, title_index).values

    adjacency_matrix = incidence_matrix.dot(incidence_matrix.T)
    n_papers = np.diag(adjacency_matrix).copy()
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix, authors, n_papers


def get_egocentric(df_data, author='Беляев Михаил'):
    author_papers = df_data[df_data.author_id_new == author].title.values

    coauthors = df_data.loc[df_data.title.isin(author_papers)]. \
        drop_duplicates('author_id_new').author_id_new.values

    author_ego = df_data.loc[(df_data.author_id_new.isin(coauthors)) \
                             & (df_data.author_id_new != author)].copy()

    adj, authors, n_papers = get_coauthors(author_ego)

    return adj, authors, n_papers


def load_json(path: str):
    """Load the contents of a json file."""
    with open(path, 'r') as f:
        return json.load(f)


def dump_json(value, path: str, *, indent: int = None):
    """Dump a json-serializable object to a json file."""
    with open(path, 'w') as f:
        return json.dump(value, f, indent=indent)


def get_friends_ids(user_id, access_token, additional_field=False):
    friends_url = \
        f'https://api.vk.com/method/friends.get?user_id={user_id}&access_token={access_token}&v=5.8'

    json_response = requests.get(friends_url).json()
    if json_response.get('error'):
        print('json_error')
        return list(), friends_url

    return json_response[u'response'], friends_url
