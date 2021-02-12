import os
from urllib.request import urlretrieve

import pandas as pd

FREMONT_URL = 'https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD'


def get_data(filename='fremont.csv', url=FREMONT_URL, force_download=False):
    """
    Download and cache fremont data

    Parameters
    ----------
    filename : string (optional)
        location to save the data
    url : string (optional)
        web location of the data
    force_download : bool (optional)
        if True, force download of data
    Returns
    -------
        data : pandas.DataFrame
            the fremont bridge data
    """
    if force_download or not os.path.exists(filename):
        # Download the data
        urlretrieve(url, './data/fremont.csv')
    # Read data from downloaded CSV file
    data = pd.read_csv('./data/fremont.csv',
                       index_col='Date',
                       parse_dates=True)
    return data