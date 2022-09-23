# Functions for scraping

from google_play_scraper import Sort, reviews_all ,reviews
import pandas as pd
import numpy as np


def get_reviews(applist):
    df_combined = []
    for name, apps in applist:
        all_reviews, _ = reviews(apps,
                        # sleep_milliseconds=0,  # defaults to 0
                          lang='en',  # defaults to 'en'
                          country='sg',  # defaults to 'us'
                          sort=Sort.NEWEST,
                          count = 10
                         )
        df = pd.DataFrame(np.array(all_reviews), columns=['review'])
        df = df.join(pd.DataFrame(df.pop('review').tolist()))
        df['app'] = name
        df['platform'] = 'android'
        columns = list(df)
        appendlist = df.values.tolist()
        for entry in appendlist:
            df_combined.append(entry)
    df_combined = pd.DataFrame(df_combined)
    df_combined.columns = columns
    df_combined = df_combined.dropna(subset=['content']).reset_index()

    return df_combined