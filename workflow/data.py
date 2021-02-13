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
    # Download the data, if necessary
    if force_download or not os.path.exists(filename):
        urlretrieve(url, './data/fremont.csv')

    # Read data from CSV file
    data = pd.read_csv('./data/fremont.csv', index_col='Date')
    # data = pd.read_csv('./data/fremont.csv', index_col='Date', parse_dates=True)
    data.columns = ['Total', 'East', 'West']

    # Convert the index format from string to datetime
    try:
        # Put in try incase data has changed
        data.index = pd.to_datetime(data.index, format='%m/%d/%Y %I:%M:%S %p')
    except TypeError:
        # Convert without specif format, may take longer but with correct result
        data.index = pd.to_datetime(data.index)

    return data