import pandas as pd
from pathlib import Path
import requests
from SPARQLWrapper import SPARQLWrapper, JSON


def logo_scraper(nb_svgs=100, target_folder='data/scraped_svgs'):
    """ SPARQL scraper to collect logos of German companies from Wikidata.

    Args:
        nb_svgs (int): Number of SVGs that are scraped.
        target_folder (str): Target directory where scraped SVGs are saved.

    Returns:
        pd.DataFrame: Dataframe containing names of scraped logos.

    """
    endpoint = 'https://query.wikidata.org/sparql'

    # query German companies
    query = f'''
            prefix wd:       <http://www.wikidata.org/entity/>
            prefix wdt:      <http://www.wikidata.org/prop/direct/>

            SELECT DISTINCT ?x ?xLabel ?xLogo
            WHERE
            {{
                ?x rdfs:label ?xLabel ;
                    wdt:P154 ?xLogo ;
                    wdt:P31 wd:Q6881511 ;
                    wdt:P17 wd:Q183 .
                FILTER regex(str(?xLogo), "svg") .
                FILTER (lang(?xLabel) = "en")
            }}
            LIMIT {nb_svgs}
            '''

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # Create dataframe containing names of scraped logos
    df = pd.DataFrame(results['results']['bindings'])
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x['value'])

    # Save scraped SVGs in target folder
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    for i in range(df.shape[0]):
        r = requests.get(df.iloc[i, 1])
        with open(f'{target_folder}/{df.iloc[i, 2]}.svg', 'wb') as f:
            f.write(r.content)

    return df


if __name__ == '__main__':
    df = logo_scraper(nb_svgs=10, target_folder='../../data/scraped_svgs')
