from time import time
import pandas as pd
from datetime import datetime
from collections import defaultdict

from .io import *
from .utils import *


def uiuc_to_db(
    folder_in=UIUC_DATASET,
    folder_out=[UIUC_DB],
    file_out=['uiuc.db'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    output=True
):
    """
    Read log files of each user, detect user_hash based on the log names
    then rename the folder to the user_id and extract data from log files
    """

    # Edit paths
    files = folder_walk(UIUC_DATASET)
    file_out = path_edit(
        file_out,
        folder_out[0],
        label_file_out,
        label_folder_out,
    )[0]

    # Dictionary {User : MAC}
    user_mac = {}
    for f in files:
        sp = f.split('/')
        # Second to last is user_id [1-28]
        u = int(sp[-2])
        # Last is the file_name = datetime + mac
        d, mac = sp[-1].split('.')
        dict_add(user_mac, u, mac)

    # Sort user_mac based on user_id (key)
    user_mac = sorted([(int(k), v) for k, v in user_mac.items()])

    # Save user_mac to DB
    query = f'''
    CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    mac TEXT NOT NULL
    );
    '''
    db_execute(file_out, query)
    query = 'INSERT INTO users VALUES (?,?)'
    db_execute_many(file_out, query, user_mac)

    # Read BT and WiFI data
    # error_list = [] # TEST
    user_b = defaultdict(list)  # Bluetooth
    user_w = defaultdict(list)  # WiFi
    b_mac = []  # Bluetooth MAC list
    w_mac = []  # WiFi MAC list

    # Add BT users to the list of bluetooth devices
    # They are not needed in WiFi
    # Because they only see access points MAC in logs
    b_mac = [u[1] for u in user_mac]

    for file in files:
        is_b = False
        is_w = False
        sp = file.split('/')
        u = int(sp[-2])
        d, mac = sp[-1].split('.')
        # Check the type of log file
        if d[0] == 's':  # BT
            is_b = True
        else:  # WiFi
            is_w = True

        # Read file line by line -> save to DF -> convert DF column to series
        df = pd.read_csv(file, names=['line'])
        lines = df.line
        last_time = ''

        for i in range(len(lines)):
            # Length of line with time info = 21
            # Length of line with mac info = 40
            if len(lines[i]) < 40:  # We read a line with time info ...
                # String format
                # last_time = lines[i]
                # Datetime format
                last_time = datetime.strptime(
                    lines[i][2:], '%m-%d-%Y %H:%M:%S'
                )
                continue
            else:  # We read a line with mac address ...
                # If last_time is out of range i.e. < 2010-2-25 then skip until read a valid one
                if last_time < datetime(2010, 2, 25):
                    continue
                # If the last_time was valid and the line is a BT or wifi do ...
                if is_b:  # Bluetooth
                    b_index = 0
                    # b_index = -1  # TEST: changed to -1 to detect the possible errors
                    if lines[i] not in b_mac:
                        # Add new MAC to list
                        b_mac.append(lines[i])
                        # Index of new mac = (len - 1) of list, becasue it was the latest added element
                        b_index = len(b_mac) - 1
                    else:
                        b_index = b_mac.index(lines[i])
                    # Add new BT log to dictionary {user:[(time,mac)]
                    user_b[u].append((last_time, b_index))
                    # TEST
                    # if b_index < 0:
                    # if last_time < datetime(2010,2,24):
                    # error_list.append((file,lines[i],last_time))

                else:  # WiFi
                    w_index = 0
                    # w_index = -1  # TEST
                    if lines[i] not in w_mac:
                        # Add new MAC to list
                        w_mac.append(lines[i])
                        # Index of new mac = (len - 1) of list becasue it is the last added element
                        w_index = len(w_mac) - 1
                    else:
                        w_index = w_mac.index(lines[i])
                    user_w[u].append((last_time, w_index))
                    # TEST
                    # if w_index < 0:
                    # if last_time < datetime(2010,2,24):
                    # error_list.append((file,lines[i],last_time))
    # TEST : save the errors
    # np.savetxt('error_list.csv', error_list, delimiter=',', fmt='%s')

    if output:
        print('# BT devices:', len(b_mac))
        print('# WiFi devices:', len(w_mac))

    # Save {BT_id : BT_MAC} to DB
    query = f'''
    CREATE TABLE IF NOT EXISTS bluetooth (
    id INTEGER PRIMARY KEY,
    mac TEXT NOT NULL
    );
    '''
    db_execute(file_out, query)
    b_mac_insert = [(i, b_mac[i]) for i in range(len(b_mac))]
    query = 'INSERT INTO bluetooth VALUES (?,?)'
    db_execute_many(file_out, query, b_mac_insert)

    # Save {WiFi_id : WiFi_MAC} to DB
    query = f'''
    CREATE TABLE IF NOT EXISTS wifi (
    id INTEGER PRIMARY KEY,
    mac TEXT NOT NULL
    );
    '''
    db_execute(file_out, query)
    w_mac_insert = [(i, w_mac[i]) for i in range(len(w_mac))]
    query = 'INSERT INTO wifi VALUES (?,?)'
    db_execute_many(file_out, query, w_mac_insert)

    # Save BT and WiFi logs to DB with following columns
    # user_node, bluetooth_node / ap_node , time, bluetooth = 0 / wifi = 1
    query = f'''
    CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY,
    user INTEGER NOT NULL,
    mac INTEGER NOT NULL,
    time TIMESTAMP,
    wifi INTEGER
    );
    '''
    db_execute(file_out, query)
    interaction_insert = []
    for k, v in user_b.items():  # k = user
        # v is a list of interactions = (time, MAC)
        for item in v:
            interaction_insert.append((k, item[1], item[0], 0))
    for k, v in user_w.items():
        for item in v:
            interaction_insert.append((k, item[1], item[0], 1))
    query = 'INSERT INTO logs(user,mac,time,wifi) VALUES (?,?,?,?)'
    db_execute_many(file_out, query, interaction_insert)

    if output:
        print('# interactions:', len(interaction_insert))


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
    Read PhoneLab dataset and create a DB file

    Scan folder has more device (=277) than connect folder (=274)
    thus we create device names from scan folder

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
        Create a new project into the projects table
        """
        users = folder_walk_folder(folder_in, name=True)
        users = list(enumerate(users, 0))
        query = """ INSERT INTO users (id,name) VALUES (?,?) """
        db_execute_many(file_out, query, users)

    def db_lookup(name, con, table="ssids", add=True):
        """
        Check if ssid (or bssid) name exist, if not, add it to table
        return the ID at the end
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
        query = f"INSERT INTO logs (user,ssid,bssid,connect,time,date,day,hour,minute) VALUES (?,?,?,?,?,?,?,?,?);"
        cur = con.cursor()
        cur.execute(
            query, (
                user, ssid, bssid, connect, time, time.date(), time.weekday(),
                time.hour, time.minute
            )
        )

    def db_complete(folder_in, file_out, connect=1):
        # i = 0
        folders = folder_walk_folder(folder_in)
        # Enter each user's folder
        for folder in folders:
            # i += 1
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
            # if i == 1:
            #     break

    # Edit paths and reading folders/files
    # files_connect = folder_walk_folder(folder_in[0])
    # files_scan = folder_walk_folder(folder_in[1])

    file_out = path_edit(
        file_out,
        folder_out[0],
        label_file_out,
        label_folder_out,
    )[0]

    # db_initialize(file_out)

    # db_add_users(folder_in[1], file_out)

    # Add connect logs to db
    # db_complete(folder_in[0], file_out)

    # Add scan logs to db
    db_complete(folder_in[1], file_out, connect=0)

    # Remove noise of DB
    # The records before 2014-11-07 (= 43 rows of connect)
    # And same for dates after 2015-04-03 (= zero rows is returned)
    # query = f"DELETE FROM logs WHERE date < '2014-11-07';"
    # db_execute(file_out, query)


def phonelab_sp(
    folder_in=[PHONELAB_DB],
    folder_out=[PHONELAB_DB],
    file_in=['phonelab.db'],
    file_out=['phonelab.db'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    output=True
):
    """
    Connect to PhoneLab database file, create spario-temporal matrix
    """
    def db_select_df(file_in, query):
        try:
            con = sqlite3.connect(file_in)
            df = pd.read_sql_query(query, con)
            # df = pd.read_sql_query(query, con, parse_dates=['date'])
        except sqlite3.Error as error:
            print(error)
        finally:
            con.close()
        return df

    # Edit paths
    file_in = path_edit(
        file_in,
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]

    file_out = path_edit(
        file_out,
        folder_out[0],
        label_file_out,
        label_folder_out,
    )[0]

    # Connect to DB
    print(file_in)

    # Calculate unique dates in DB
    query = f'SELECT DISTINCT date FROM logs ORDER BY date;'
    df = db_select_df(file_in, query)
    times = pd.to_datetime(df['date'])

    query = f'SELECT DISTINCT id FROM ssids ORDER BY id;'
    df = db_select_df(file_in, query)
    ssids = list(df['id'].index)  # 1176 SSID was detected

    # Cycle through each date, read logs
    times = [times[0]]  # Test
    for t in times:
        print('date:', t.date())
        query = f'SELECT user,ssid,time FROM logs WHERE date = \'{t.date()}\' ORDER BY date;'
        df = db_select_df(file_in, query)
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].apply(lambda x: x.replace(minute=0, second=0))
        print(df)
