import pandas as pd
from scipy import sparse
import time
from datetime import datetime
from itertools import permutations
from pprint import pprint

from utils.paths import *
from utils.helpers import *


def phonelab_to_db(
    folder_in=[PHONELAB_CONNECT, PHONELAB_SCAN],
    folder_out=[PHONELAB_DB],
    file_out=['phonelab.db'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    output=True
):
    """
    Read PhoneLab dataset and create a database

    """
    def db_initialize(file_out):
        """
        Create database and its tables
        """
        # Users
        query = """ CREATE TABLE IF NOT EXISTS users (id integer PRIMARY KEY, name text NOT NULL); """
        db_execute(file_out, query)
        # SSID
        query = """ CREATE TABLE IF NOT EXISTS ssids (id integer PRIMARY KEY, name text NOT NULL); """
        db_execute(file_out, query)
        # BSSID
        query = """ CREATE TABLE IF NOT EXISTS bssids (id integer PRIMARY KEY, name text NOT NULL); """
        db_execute(file_out, query)
        # Zones (SSID -> BSSID)
        query = """ CREATE TABLE IF NOT EXISTS zones (id integer PRIMARY KEY, ssid integer NOT NULL, bssid integer NOT NULL, FOREIGN KEY (ssid) REFERENCES ssids (id), FOREIGN KEY (bssid) REFERENCES bssids (id)); """
        db_execute(file_out, query)
        # Logs
        query = """ CREATE TABLE IF NOT EXISTS logs (id integer PRIMARY KEY, user integer NOT NULL, ssid integer NOT NULL, bssid integer NOT NULL, connect integer NOT NULL,time integer NOT NULL, date text NOT NULL, day integer NOT NULL, hour integer NOT NULL, minute integer NOT NULL, FOREIGN KEY (user) REFERENCES users (id), FOREIGN KEY (ssid) REFERENCES ssids (id), FOREIGN KEY (bssid) REFERENCES bssids (id)); """
        db_execute(file_out, query)

    def db_add_users(folder_in, file_out):
        """
        Read user / device names from dataset folder
        Then add them into users table
        """
        users = folder_walk_folder(folder_in, name=True)
        users = list(enumerate(users, 0))
        query = """ INSERT INTO users (id,name) VALUES (?,?) """
        db_execute_many(file_out, query, users)

    def db_lookup(name, con, table="ssids", add=True):
        """
        Check if ssid (or bssid) exist, if not, add it to related table
        Then return the ID of ssid (or bssid) at the end
        """
        result = 0
        action = ""
        query = f"SELECT * FROM {table} WHERE name = ? LIMIT 1;"
        cur = con.cursor()
        # Look up the name of ssid (or bssid)
        cur.execute(query, (name, ))
        row = cur.fetchone()
        if row is not None:
            result = row[0]  # ID
            action = "found"
        else:
            if add:
                # Not found, add to DB
                query = f"INSERT INTO {table} (name) VALUES(?);"
                cur.execute(query, (name, ))
                con.commit()
                result = cur.lastrowid
                action = "added"
        cur.close()
        return (result, action)

    def db_zone(ssid, bssid, con):
        """
        Check the zone table for ssid -> bssid entry
        If missing, add it to the table
        """
        query = f"SELECT * FROM zones WHERE ssid = ? AND bssid = ? LIMIT 1;"
        cur = con.cursor()
        # Look up the name of ssid -> bssid entry
        cur.execute(query, (ssid, bssid))
        row = cur.fetchone()
        if row is None:
            # Not found, add to DB
            query = f"INSERT INTO zones (ssid,bssid) VALUES(?,?);"
            cur.execute(query, (ssid, bssid))
            con.commit()
        cur.close()

    def db_add_log(user, ssid, bssid, connect, time: datetime, con):
        """
        Add the interaction between user and AP device
        """
        query = f"INSERT INTO logs (user,ssid,bssid,connect,time,date,day,hour,minute) VALUES (?,?,?,?,?,?,?,?,?);"
        cur = con.cursor()
        cur.execute(
            query, (
                user, ssid, bssid, connect, time, time.date(), time.weekday(),
                time.hour, time.minute
            )
        )

    def db_complete(folder_in, file_out, connect=1):
        folders = folder_walk_folder(folder_in)
        # In case of error, instead of starting from beginning
        # Enter the user id number to resume creating database from there
        # user_id = 256
        query = f'SELECT name FROM users WHERE id >= {user_id} ORDER BY id;'
        folders = db_select_df(file_out, query)['name']
        folders = [os.path.join(folder_in, f) for f in folders]
        for folder in folders:
            try:
                con = sqlite3.connect(file_out)
                # Extract user name and if from folder name
                user = folder.split('/')[-1]
                user_id = db_lookup(user, con, 'users', False)[0]
                print(user_id, ':', user)
                # Read every file
                for file_path in folder_walk(folder):
                    with open(file_path) as f:
                        # Analyze each line
                        for line in f:
                            # Convert line to dictionary
                            line_dict = json.loads(line)
                            # Only add connection if connected flag is true
                            connect_flag = line_dict.get('connected', False)
                            if connect_flag or connect == 0:
                                # Check if SSID has been observed before
                                # If yes, get the id, if not add and get id
                                ssid_id = db_lookup(line_dict['SSID'], con)[0]
                                bssid_id = db_lookup(
                                    line_dict['BSSID'], con, table='bssids'
                                )[0]
                                db_zone(ssid_id, bssid_id, con)
                                t = datetime.utcfromtimestamp(
                                    int(line_dict['timestamp'])
                                )
                                db_add_log(
                                    user_id, ssid_id, bssid_id, connect, t, con
                                )
                # Commit the changes
                con.commit()
            except sqlite3.Error as e:
                print(e)
            finally:
                con.close()

    # Edit paths and reading folders and files inside them
    files_connect = folder_walk_folder(folder_in[0])
    files_scan = folder_walk_folder(folder_in[1])
    file_out = path_edit(
        file_out,
        folder_out[0],
        label_file_out,
        label_folder_out,
    )[0]

    # Create database, (un)comment if necessary
    # db_initialize(file_out)
    # Add users to users table
    # db_add_users(folder_in[1], file_out)
    # Add connect logs to db
    # db_complete(folder_in[0], file_out)
    # Add scan logs to db
    # db_complete(folder_in[1], file_out, connect=0)
    # Remove noise of DB
    # The records before 2014-11-07 or after 2015-04-03
    # query = f"DELETE FROM logs WHERE date > '2015-04-03' OR date < '2014-11-07';"
    # db_execute(file_out, query)


def phonelab_spatial(
    folder_in=[PHONELAB_DB],
    folder_out=[PHONELAB_DATA],
    file_in=['phonelab.db'],
    file_out=[
        'user_spatial_connect.npz',
        'user_day_spatial_connect.npz',
        'user_day_spatial_connect_label.csv',
        'user_day_spatial_connect_label_dow.csv',
        'user_spatial_connect_network.csv',
        'user_spatial_connect_network_weighted.csv',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    output=True
):
    """
    Create spatial matrix
    """
    # Input files i.e. Database
    file_in = path_edit(
        file_in,
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    if output:
        print('Input files:')
        print(file_in)

    # Output files
    file_out = path_edit(
        file_out,
        folder_out[0],
        label_file_out,
        label_folder_out,
    )
    if output:
        print('Output files:')
        print(file_out)

    # Read dates from logs table
    query = f'SELECT DISTINCT date FROM logs WHERE connect = 1 ORDER BY date;'
    df = db_select_df(file_in, query)
    dates = pd.to_datetime(df['date'])
    if output: print('Number of days =', len(dates))

    # Read ssids from logs table
    query = f'SELECT DISTINCT ssid FROM logs WHERE connect = 1 ORDER BY ssid;'
    df = db_select_df(file_in, query)
    ssids = list(df['ssid'])
    if output: print('Number of SSIDs =', len(ssids))

    # Create SSID-Name -> SSID-ID dictionary
    ssids = {sName: sId for sId, sName in enumerate(ssids)}

    # Read users from logs table
    query = f'SELECT DISTINCT user FROM logs WHERE connect = 1 ORDER BY user;'
    df = db_select_df(file_in, query)
    users = list(df['user'])

    # Create UserName -> UserID dictionary
    users = {uName: uId for uId, uName in enumerate(users)}

    # Spatial matrix
    M = np.zeros((len(users), len(ssids)))
    M_day = np.empty((0, len(ssids)))

    # Create ML labels
    M_day_label = []  # User of the day
    M_day_label_dow = []  # Day of the week

    # Columns of USER-USER network
    column_u = []
    column_v = []
    column_t = []  # TIME is DATE of connections
    column_l = []  # SSID that enabled the User-User egde

    # Cycle through each date and read logs then create ...
    for t in dates:
        if output:
            print('Date =', t.date())
        # Select USER-SSID for each date
        query = f'SELECT user,ssid FROM logs WHERE date = \'{t.date()}\';'
        df = db_select_df(file_in, query)
        day_users = df['user'].unique()
        # Create multiple rows based on number of unique users
        M_day_dict = {u: np.zeros((1, len(ssids))) for u in day_users}
        for key, group_user_ssid in df.groupby(['ssid']):
            group_users = group_user_ssid['user'].unique()
            for user in group_users:
                # Update matrix M
                M[users[user]][ssids[key]] += 1
                # Also update matrix M-Day
                M_day_dict.get(user)[0][ssids[key]] += 1
            if len(group_users) > 1:
                # Update User-User network
                connected_users = list(permutations(group_users, 2))
                for element in connected_users:
                    column_u.append(element[0])
                    column_v.append(element[1])
                    column_t.append(t.date())
                    column_l.append(key)
                # TODO the network could have repeated edges due to users ...
                # Connecting via different SSID in one day, which can be cleaned
                # Later convert to proper weight
        for user in M_day_dict:
            M_day = np.append(M_day, M_day_dict[user], axis=0)
            M_day_label.append(user)
            M_day_label_dow.append(t.weekday())
    # Check if any of matrix's rows has all zero values
    # print(np.where(~M.any(axis=1)))
    # Save the sparse version of matrix M
    sparse.save_npz(file_out[0], sparse.csr_matrix(M))
    # Save the sparse version of matrix M-Day
    sparse.save_npz(file_out[1], sparse.csr_matrix(M_day))
    # Save labels
    np.savetxt(file_out[2], M_day_label, delimiter=',', fmt='%s')
    np.savetxt(file_out[3], M_day_label_dow, delimiter=',', fmt='%s')
    # Dataframe of user-user network
    df_network = pd.DataFrame(
        list(zip(column_u, column_v, column_t, column_l)),
        columns=['u', 'v', 't', 'l']
    )
    df_network.to_csv(file_out[4], index=False)
    # Remove duplicate edges
    df_network = df_network.drop_duplicates()
    # Drop location of connections
    df_network.drop(columns=['l'], inplace=True)
    # Covert duplicated edges to weight
    df_network = df_network.groupby(
        df_network.columns.tolist()
    ).size().reset_index(name='w')  # Weight of edges as 'w'
    df_network.to_csv(file_out[5], index=False)


def phonelab_spatial_temporal(
    folder_in=[PHONELAB_DB],
    folder_out=[PHONELAB_DATA],
    file_in=['phonelab.db'],
    file_out=[
        'user_spatial_temporal_connect.npz',
        'user_day_spatial_temporal_connect.npz',
        'user_day_spatial_temporal_connect_label.csv',
        'user_day_spatial_temporal_connect_label_dow.csv',
        'user_spatial_temporal_connect_network.csv',
        'user_spatial_temporal_connect_network_weighted.csv',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    connect=1,
    output=True
):
    """
    Create spario-temporal matrix : USER -> SSID x HOUR
    """
    # Paths
    file_in = path_edit(
        file_in,
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    if output:
        print('Reading the input file:')
        print(file_in)
    file_out = path_edit(
        file_out,
        folder_out[0],
        label_file_out,
        label_folder_out,
    )
    # Calculate unique dates in DB
    query = f'SELECT DISTINCT date FROM logs WHERE connect = {connect} ORDER BY date;'
    df = db_select_df(file_in, query)
    dates = pd.to_datetime(df['date'])
    if output: print('Days =', len(dates))
    # Calculate unique ssid (locations) in DB
    query = f'SELECT DISTINCT ssid FROM logs WHERE connect = {connect} ORDER BY ssid;'
    df = db_select_df(file_in, query)
    ssids = list(df['ssid'])  # 1176 SSID was detected
    # Create SSID-Name -> SSID-ID dictionary
    ssids = {sName: sId for sId, sName in enumerate(ssids)}
    if output: print('SSIDs =', len(ssids))
    # Calculate unique users in DB
    query = f'SELECT DISTINCT user FROM logs WHERE connect = {connect} ORDER BY user;'
    df = db_select_df(file_in, query)
    users = list(df['user'])  # 270 User was detected (total is 277)
    # Create UserName -> UserID dictionary
    users = {uName: uId for uId, uName in enumerate(users)}
    if output: print('Users =', len(users))
    # Spatio-temporal matrix of USER & SSID x TIMES (or 24 hours)
    M = np.zeros((len(users), len(ssids) * 24))
    # Spatio-temporal matrix USER at each day -> SSID x TIMES
    # Each row denotes activity of a user in one particular day
    M_day = np.empty((0, len(ssids) * 24))
    M_day_label = []
    M_day_label_dow = []  # Day of the week
    # Three columns for DF of USER-USER multi-network
    # Edge exist if two users contact at least one SSID in one-hour time window
    # Result is a network wtih 3856776 edges (after weighting repeated edges)
    column_u = []
    column_v = []
    column_t = []
    # Cycle through each date and analyze the logs ...
    for t in dates:
        if output:
            print('Date:', t.date())
        query = f'SELECT user,ssid,time FROM logs WHERE connect = {connect} AND date = \'{t.date()}\' ORDER BY time;'
        # Do a SQL select query
        df = db_select_df(file_in, query)
        # Covert time column to datetime format and the right time scale
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].apply(lambda x: x.replace(minute=0, second=0))
        # Extract unique active users in that date
        day_users = df['user'].unique()
        # Create a dict/row of data for each active user
        M_day_dict = {u: np.zeros((1, len(ssids) * 24)) for u in day_users}
        # Find users connect to same SSID in underlying date
        for key, group_user_ssid in df.groupby(['time', 'ssid']):
            group_users = group_user_ssid['user'].unique()
            for user in group_users:
                M[users[user]][ssids[key[1]] * 24 + key[0].hour] += 1
                # Also update M-Day matrix
                M_day_dict.get(user)[0][ssids[key[1]] * 24 + key[0].hour] += 1
            if len(group_users) > 1:
                connected_users = list(permutations(group_users, 2))
                for element in connected_users:
                    column_u.append(element[0])
                    column_v.append(element[1])
                    column_t.append(key[0])
        for user in M_day_dict:
            M_day = np.append(M_day, M_day_dict[user], axis=0)
            M_day_label.append(user)
            M_day_label_dow.append(t.weekday())
    # Save the sparse matrix
    sparse.save_npz(file_out[0], sparse.csr_matrix(M))
    # Save User-Day sparse matrix
    sparse.save_npz(file_out[1], sparse.csr_matrix(M_day))
    # Save labels
    np.savetxt(file_out[2], M_day_label, delimiter=',', fmt='%s')
    np.savetxt(file_out[3], M_day_label_dow, delimiter=',', fmt='%s')
    # Dataframe of user-user network
    df_network = pd.DataFrame(
        list(zip(column_u, column_v, column_t)), columns=['u', 'v', 't']
    )
    # Save dataframe as network edge list
    df_network.to_csv(file_out[4], index=False)
    # Remove duplicate edges
    df_network = df_network.drop_duplicates()
    # Covert duplicated edges to weight
    df_network = df_network.groupby(df_network.columns.tolist()
                                    ).size().reset_index(name='w')
    df_network.to_csv(file_out[5], index=False)


def phonelab_selected_ssid(
    folder_in=[PHONELAB_DATA],
    folder_out=[PHONELAB_DATA],
    file_in=['user_spatial_connect.npz'],
    file_out=['selected_ssid.csv'],
    label_folder_in='spatial',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    selected=2,
    share=False,
    output=True
):
    """
    Create spatial matrix considering selected SSIDs
    """
    # Input files i.e. Database
    file_in = path_edit(
        file_in,
        folder_in[0] + '/selected_0',
        label_file_in,
        label_folder_in,
    )
    if output:
        print('Input files:')
        print(file_in)

    # Output files
    selected_lable = selected
    if share: selected_lable = 1
    file_out1 = path_edit(
        file_out,
        folder_out[0] + f'/selected_{selected_lable}',
        label_file_out,
        'spatial',
    )
    file_out2 = path_edit(
        file_out,
        folder_out[0] + f'/selected_{selected_lable}',
        label_file_out,
        'spatial_temporal',
    )
    file_out = file_out1 + file_out2
    if output:
        print('Output files:')
        pprint(file_out)

    # Load user spatial matrix from file
    M = sparse.load_npz(file_in[0]).toarray()

    # Important SSID extracted users
    imp_ssids = []
    imp_ssids_users = {}

    # Cycle through rows or matrix or user connectivity ...
    for idx, row in enumerate(M):

        # Change type
        row = row.astype(int)
        # Number of non-zero elements
        connected_ssids = np.where(row > 0)[0]

        # Check if only single user connected to analyzing ssid
        if len(connected_ssids) == 1:
            imp_ssids.append(connected_ssids[0])
            if connected_ssids[0] not in imp_ssids_users:
                imp_ssids_users[connected_ssids[0]] = [idx]
            else:
                imp_ssids_users[connected_ssids[0]].append(idx)
            # if output: print({idx}, '->', *connected_ssids)

        # Case that user connected to more than one ssid
        if len(connected_ssids) > 1:
            row_sort_idx = np.argsort(row)
            # Pick only top <selected> APs
            for ssid in row_sort_idx[::-1][:selected]:
                imp_ssids.append(ssid)
                if ssid not in imp_ssids_users:
                    imp_ssids_users[ssid] = [idx]
                else:
                    imp_ssids_users[ssid].append(idx)

    # Convert and sort important ssid list while removing duplicates
    imp_ssids = sorted(list(set(imp_ssids)))

    if share:
        # Convert M to the matrix of zeros and ones only
        M[M > 0] = 1

        # Calculate highly connected SSIDs
        # Using sum of columns of normalized M
        ssid_dist = np.array(np.sum(M, axis=0, dtype=int).reshape(1, -1))[0]
        ssid_dist_tuple = sorted(
            list(enumerate(ssid_dist)), key=lambda e: e[1], reverse=True
        )
        # Shared elements between important ssid and highly connected ones
        imp_ssid_inter = set(imp_ssids).intersection(
            list(zip(*ssid_dist_tuple[:len(imp_ssids)]))[0]
        )
        # Sort one more time
        imp_ssids = sorted(list(imp_ssid_inter))

    # Output result stat
    if output:
        print(
            f'{len(imp_ssids)} important SSIDs has been selected by taking top {selected} APs from each user.'
        )

    # Save important ssid list
    np.savetxt(file_out[0], imp_ssids, delimiter=',', fmt='%s')
    np.savetxt(file_out[1], imp_ssids, delimiter=',', fmt='%s')


def phonelab_spatial_selected(
    folder_in=[PHONELAB_DB],
    folder_out=[PHONELAB_DATA],
    file_in=['phonelab.db'],
    file_out=[
        'user_spatial_connect.npz',
        'user_day_spatial_connect.npz',
        'user_day_spatial_connect_label.csv',
        'user_day_spatial_connect_label_dow.csv',
        'user_spatial_connect_network.csv',
        'user_spatial_connect_network_weighted.csv',
        'selected_ssid.csv',
        'users.csv',
    ],
    label_folder_in='',
    label_folder_out='spatial',
    label_file_in='',
    label_file_out='',
    selected=0,
    output=True
):
    """
    Create spatial matrix considering selected SSIDs
    """
    # Input files i.e. Database
    file_in = path_edit(
        file_in,
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    if output:
        print('Input files:')
        print(file_in)

    # Output files
    file_out = path_edit(
        file_out,
        folder_out[0] + f'/selected_{selected}',
        label_file_out,
        label_folder_out,
    )
    if output:
        print('\nOutput files:')
        pprint(file_out)

    # Read dates from logs table
    query = f'SELECT DISTINCT date FROM logs WHERE connect = 1 ORDER BY date;'
    df = db_select_df(file_in, query)
    dates = pd.to_datetime(df['date'])
    if output: print('\nNumber of days =', len(dates))

    # Read ssids from logs table
    query = f'SELECT DISTINCT ssid FROM logs WHERE connect = 1 ORDER BY ssid;'
    df = db_select_df(file_in, query)
    ssids_all = list(df['ssid'])
    if output: print('Number of SSIDs =', len(ssids_all))

    # Read selected ssids from the file
    ssids = pd.read_csv(
        file_out[6], index_col=False, header=None, names=['ssid']
    )['ssid'].astype(int).values
    if output: print('Number of selected SSIDs =', len(ssids))

    # Create SSID-Name -> SSID-ID dictionary
    ssids = {sName: sId for sId, sName in enumerate(ssids)}

    # Read users from logs table
    query = f'SELECT DISTINCT user FROM logs WHERE connect = 1 ORDER BY user;'
    df = db_select_df(file_in, query)
    users = list(df['user'])
    if output: print('Number of users =', len(users))

    # Create UserName -> UserID dictionary
    users = {uName: uId for uId, uName in enumerate(users)}

    # Spatial matrix
    M = np.zeros((len(users), len(ssids)))
    M_day = np.empty((0, len(ssids)))

    # Create ML labels
    M_day_label = []  # User of the day
    M_day_label_dow = []  # Day of the week

    # Columns of USER-USER network
    column_u = []
    column_v = []
    column_t = []  # TIME is DATE of connections
    column_l = []  # SSID that enabled the User-User egde

    # Cycle through each date and read logs then create ...
    for t in dates:
        if output:
            if t == dates.iloc[0]:
                print('\nDates:')
            print(t.date(), end=', ')
            if t == dates.iloc[-1]:
                print('\n')

        # Select USER-SSID for each date
        query = f'SELECT user,ssid,time FROM logs WHERE connect = 1 AND date = \'{t.date()}\' ORDER BY time;'
        df = db_select_df(file_in, query)

        # Active users in selected date
        day_users = df['user'].unique()

        # Create multiple rows based on number of unique users
        M_day_dict = {u: np.zeros((1, len(ssids))) for u in day_users}

        # Find users connecting to same SSID in underlying date
        for key, group_user_ssid in df.groupby(['ssid']):

            # Only process if SSID is in the selected SSID set
            if key in ssids:
                group_users = group_user_ssid['user'].unique()
                for user in group_users:

                    # Update matrix M
                    M[users[user]][ssids[key]] += 1

                    # Also update matrix M-Day
                    M_day_dict.get(user)[0][ssids[key]] += 1

                # Update User-User network
                if len(group_users) > 1:
                    connected_users = list(permutations(group_users, 2))
                    for element in connected_users:
                        column_u.append(element[0])
                        column_v.append(element[1])
                        column_t.append(t.date())
                        column_l.append(key)

        for user in M_day_dict:
            # Check if row of dictionary is not all zero elements
            if np.any(M_day_dict[user]):
                M_day = np.append(M_day, M_day_dict[user], axis=0)
                M_day_label.append(user)
                M_day_label_dow.append(t.weekday())

    # Check if any of matrix's rows has all zero values
    zero_rows = np.where(~M.any(axis=1)[0])[0]
    # zero_rows = np.append(zero_rows, 1)  # Test
    if len(zero_rows > 0):
        if output:
            print(f'{len(zero_rows)} number of rows in M are all zeros:')
            print(zero_rows)
        for row in zero_rows:
            # Delete all-zero row from M
            M = np.delete(M, row, axis=0)
        # Delete from user list
        users = {k: v for k, v in users.items() if v not in zero_rows}
        print('New number of users =', len(users))

    # Save active users
    np.savetxt(file_out[7], sorted(users.keys()), delimiter=',', fmt='%s')

    # Check if any of matrix's rows has all zero values
    zero_rows = np.where(~M_day.any(axis=1)[0])[0]
    if len(zero_rows > 0):
        if output and len(zero_rows) > 1:
            print(f'{len(zero_rows)} number of rows in M-Day are all zeros:')
            print(zero_rows)
        for row in zero_rows:
            # Delete all-zero row from M-Day
            M_day = np.delete(M_day, row, axis=0)
            # Delete from label lists
            del M_day_label[row]
            del M_day_label_dow[row]

    # Save sparse matrices
    sparse.save_npz(file_out[0], sparse.csr_matrix(M))
    sparse.save_npz(file_out[1], sparse.csr_matrix(M_day))
    if output:
        print('Saving spatial matrices:')
        print(f'\t- {M.shape}')
        print(f'\t- {M_day.shape}')

    # Save labels
    np.savetxt(file_out[2], M_day_label, delimiter=',', fmt='%s')
    np.savetxt(file_out[3], M_day_label_dow, delimiter=',', fmt='%s')

    # Create dataframe for user-user network
    df_network = pd.DataFrame(
        list(zip(column_u, column_v, column_t, column_l)),
        columns=['u', 'v', 't', 'l']
    )

    # Save user-user network edges
    df_network.to_csv(file_out[4], index=False)

    # Remove duplicate edges
    df_network = df_network.drop_duplicates()

    # Drop location of connections
    df_network.drop(columns=['l'], inplace=True)

    # Convert duplicated rows as weighted edges with label W
    df_network = df_network.groupby(df_network.columns.tolist()
                                    ).size().reset_index(name='w')

    # Save weighted user-user network edges
    df_network.to_csv(file_out[5], index=False)


def phonelab_spatial_temporal_selected(
    folder_in=[PHONELAB_DB],
    folder_out=[PHONELAB_DATA],
    file_in=['phonelab.db'],
    file_out=[
        'user_spatial_temporal_connect.npz',
        'user_day_spatial_temporal_connect.npz',
        'user_day_spatial_temporal_connect_label.csv',
        'user_day_spatial_temporal_connect_label_dow.csv',
        'user_spatial_temporal_connect_network.csv',
        'user_spatial_temporal_connect_network_weighted.csv',
        'selected_ssid.csv'
    ],
    label_folder_in='',
    label_folder_out='spatial_temporal',
    label_file_in='',
    label_file_out='',
    selected=0,
    output=True
):
    """
    Create spatial-temporal matrix considering selected SSIDs
    """
    # Input files i.e. Database
    file_in = path_edit(
        file_in,
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    if output:
        print('Input files:')
        print(file_in)

    # Output files
    file_out = path_edit(
        file_out,
        folder_out[0] + f'/selected_{selected}',
        label_file_out,
        label_folder_out,
    )
    if output:
        print('\nOutput files:')
        pprint(file_out)

    # Read dates from logs table
    query = f'SELECT DISTINCT date FROM logs WHERE connect = 1 ORDER BY date;'
    df = db_select_df(file_in, query)
    dates = pd.to_datetime(df['date'])
    if output: print('\nNumber of days =', len(dates))

    # Read ssids from logs table
    query = f'SELECT DISTINCT ssid FROM logs WHERE connect = 1 ORDER BY ssid;'
    df = db_select_df(file_in, query)
    ssids_all = list(df['ssid'])
    if output: print('Number of SSIDs =', len(ssids_all))

    # Read selected ssids from the file
    ssids = pd.read_csv(
        file_out[6], index_col=False, header=None, names=['ssid']
    )['ssid'].astype(int).values
    if output: print('Number of selected SSIDs =', len(ssids))

    # Create SSID-Name -> SSID-ID dictionary
    ssids = {sName: sId for sId, sName in enumerate(ssids)}

    # Read users from logs table
    query = f'SELECT DISTINCT user FROM logs WHERE connect = 1 ORDER BY user;'
    df = db_select_df(file_in, query)
    users = list(df['user'])
    if output: print('Number of users =', len(users))

    # Create UserName -> UserID dictionary
    users = {uName: uId for uId, uName in enumerate(users)}

    # Spatial-temporal matrix
    M = np.zeros((len(users), len(ssids) * 24))
    M_day = np.empty((0, len(ssids) * 24))

    # Create ML labels
    M_day_label = []  # User of the day
    M_day_label_dow = []  # Day of the week

    # Columns of USER-USER network
    column_u = []
    column_v = []
    column_t = []

    # Cycle through each date and analyze the logs ...
    for t in dates:
        if output:
            if t == dates.iloc[0]:
                print('\nDates:')
            print(t.date(), end=', ')
            if t == dates.iloc[-1]:
                print('\n')

        # Select USER-SSID for each date
        query = f'SELECT user,ssid,time FROM logs WHERE connect = 1 AND date = \'{t.date()}\' ORDER BY time;'
        df = db_select_df(file_in, query)

        # Covert time column to datetime format and the right time scale
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].apply(lambda x: x.replace(minute=0, second=0))

        # Extract active users in underlying date
        day_users = df['user'].unique()

        # Create a dict/row of data for each active user
        M_day_dict = {u: np.zeros((1, len(ssids) * 24)) for u in day_users}

        # Find users connect to same SSID in underlying date
        for key, group_user_ssid in df.groupby(['time', 'ssid']):

            # Process the data if SSID is among the selected SSID
            if key[1] in ssids:
                group_users = group_user_ssid['user'].unique()
                for user in group_users:

                    # Update matrix M
                    M[users[user]][ssids[key[1]] * 24 + key[0].hour] += 1

                    # Also update M-Day matrix
                    M_day_dict.get(user)[0][ssids[key[1]] * 24 +
                                            key[0].hour] += 1

                # Update User-User network
                if len(group_users) > 1:
                    connected_users = list(permutations(group_users, 2))
                    for element in connected_users:
                        column_u.append(element[0])
                        column_v.append(element[1])
                        column_t.append(key[0])

        for user in M_day_dict:
            if np.any(M_day_dict[user]):
                M_day = np.append(M_day, M_day_dict[user], axis=0)
                M_day_label.append(user)
                M_day_label_dow.append(t.weekday())

    # Check if any of matrix's rows has all zero values
    zero_rows = np.where(~M.any(axis=1)[0])[0]
    # zero_rows = np.append(zero_rows, 1)  # Test
    if len(zero_rows > 0):
        if output:
            print(f'{len(zero_rows)} number of rows in M are all zeros:')
            print(zero_rows)
        for row in zero_rows:
            # Delete all-zero row from M
            M = np.delete(M, row, axis=0)
        # Delete from user list
        users = {k: v for k, v in users.items() if v not in zero_rows}
        print('New number of users =', len(users))

    # Save active users
    np.savetxt(file_out[7], sorted(users.keys()), delimiter=',', fmt='%s')

    # Check if any of matrix's rows has all zero values
    zero_rows = np.where(~M_day.any(axis=1)[0])[0]
    if len(zero_rows > 0):
        if output and len(zero_rows) > 1:
            print(f'{len(zero_rows)} number of rows in M-Day are all zeros:')
            print(zero_rows)
        for row in zero_rows:
            # Delete all-zero row from M-Day
            M_day = np.delete(M_day, row, axis=0)
            # Delete from label lists
            del M_day_label[row]
            del M_day_label_dow[row]

    # Save sparse matrices
    sparse.save_npz(file_out[0], sparse.csr_matrix(M))
    sparse.save_npz(file_out[1], sparse.csr_matrix(M_day))
    if output:
        print('Saving spatial matrices:')
        print(f'\t- {M.shape}')
        print(f'\t- {M_day.shape}')

    # Save labels
    np.savetxt(file_out[2], M_day_label, delimiter=',', fmt='%s')
    np.savetxt(file_out[3], M_day_label_dow, delimiter=',', fmt='%s')

    # Create dataframe for user-user network
    df_network = pd.DataFrame(
        list(zip(column_u, column_v, column_t)), columns=['u', 'v', 't']
    )

    # Save dataframe as network edge list
    df_network.to_csv(file_out[4], index=False)

    # Remove duplicate edges
    df_network = df_network.drop_duplicates()

    # Covert duplicated edges to weight
    df_network = df_network.groupby(df_network.columns.tolist()
                                    ).size().reset_index(name='w')

    # Save weighted user-user network edges
    df_network.to_csv(file_out[5], index=False)

