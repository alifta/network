# ------
# Import
# ------

import networkx as nx

from collections import Counter
from collections import defaultdict
from itertools import permutations
from networkx.algorithms import distance_measures

import scipy.stats as stats
import numpy.linalg as linalg

import math
import timeit
import powerlaw

from sklearn.preprocessing import normalize
from sklearn.preprocessing import QuantileTransformer

from .io import *
from .utils import *

# ----------------
# Temporal Network
# ----------------


def temporal_bt(
    folder_in=[FILE, DB],
    folder_out=[NETWORK],
    file_in=['bt.csv', 'uiuc.db'],
    file_out=[
        'bt_temporal_network.gpickle',
        'bt_bipartite_network.gpickle',
        'bt_temporal_edgelist.csv',
        'bt_bipartite_edgelist.csv',
        'bt_temporal_times.csv',
        'bt_bipartite_times.csv',
        'bt_temporal_nodes.csv',
        'bt_bipartite_nodes.csv',
        'bt_temporal_weights.csv',
        'bt_bipartite_weights.csv',
        'bt_temporal_weights_scaled.csv',
        'bt_bipartite_weights_scaled.csv',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    output_times=False,
    output_network=True,
    plot_weights=False,
    save_times=True,
    save_nodes=True,
    save_weights=True,
    save_network_db=True,
    save_network_csv=True,
    save_network_file=True
):
    """
    Read dataset CSV file that has (at least) 3 columns of (node-1, node-2, timestamp)
    and create temporal network which is a multiple-edge directed (aggregated) temporal graph
    with time-labeled edges of (u,v,t)
    """

    # Edit paths
    path1 = path_edit(
        [file_in[0]],
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    path2 = path_edit(
        [file_in[1]],
        folder_in[1],
        label_file_in,
        label_folder_in,
    )[0]
    file_out = path_edit(
        file_out,
        folder_out[0],
        label_file_out,
        label_folder_out,
    )

    # G is user-user network
    graph = nx.MultiDiGraph()
    graph.name = 'Tremporal Network'
    # B is user-other-devices biparite network
    bipartite = nx.MultiDiGraph()
    bipartite.name = 'Bipartite Tremporal Network'

    # Read dataset
    data = pd.read_csv(
        path1,
        header=None,
        names=['user', 'mac', 'time'],
        parse_dates=['time']
    )

    # Print times distributions
    if output_times:
        times = pd.Series(sorted(data.time.unique()))
        print(f'Number of unique timestamps = {len(times)}')
        print('Timestamp : Frequency')
        for t_size in data.groupby('time').size().iteritems():
            print(
                f'{times[times == t_size[0]].index[0]}) {t_size[0]} : {t_size[1]}'
            )
        print()

    # Create timestamp list
    times = []
    times_bipartite = []

    # Dictionary {time:{(user-1,user-2):weight}}
    time_edges = defaultdict()  # User -> User
    time_bipartite_edges = defaultdict()  # Users -> Others

    # Group interactions by time filtering
    for key_time, group_user_mac in data[['user',
                                          'mac']].groupby(data['time']):
        # Co-location graph edges in filtered timestamp
        temp_edges = []
        for key_mac, group_connection in group_user_mac.groupby(['mac']
                                                                )['user']:
            # Users of connecting to filtered MAC
            temp_users = list(group_connection.unique())
            # If the ID of shared connected mac is [0-27]
            # Users directly connect to another bluetooth user
            if key_mac < 28:
                # Case where only one direct connection: user -> user
                if len(temp_users) <= 1:
                    temp_edges.append((temp_users[0], key_mac))
                    # Comment next line, if wanna have directed edges
                    temp_edges.append((key_mac, temp_users[0]))
                else:
                    # Case where more than 1 user undirectly connect together via 1 direct user -> user edge
                    # Direct edges
                    for element in temp_users:
                        temp_edges.append((element, key_mac))
                        # Uncomment next line, if wanna have undirected edges when one user observe another user directly
                        temp_edges.append((key_mac, element))
                    # Undirect edges
                    connected_users = list(permutations(temp_users, 2))
                    connected_users = [tuple(e) for e in connected_users]
                    temp_edges.extend(connected_users)
            # If users are connected to device with ID > 28
            # Meaning indirect edges with each other
            else:
                # Only consider cases of more than 1 unique user for co-location indirected edges
                if len(temp_users) > 1:
                    # Undirect edges
                    connected_users = list(permutations(temp_users, 2))
                    connected_users = [tuple(e) for e in connected_users]
                    temp_edges.extend(connected_users)
        # Add edges of current timestamp (with their strength) to dictionary
        if len(temp_edges) > 0:
            time_edges[key_time] = dict(Counter(temp_edges))
        # Bipartite graph edges
        # We don't care about MAC < 28, just want to count
        # How many times in each timestamp a user connect to a MAC
        # Dictionary {time:{(user,mac):weight}}
        bipartite_edges = {}
        # Filter connections based on (user -> mac) interaction and its weight
        for key_mac, group_connection in group_user_mac.groupby(
            ['mac', 'user']
        ):
            # User connected to filtered MAC with X number of times
            bipartite_edges[key_mac] = len(group_connection)
        # Add edges of this time (with their strength) to dictionary
        time_bipartite_edges[key_time] = bipartite_edges

    # Co-location network data
    l1, l2, l3, l4 = [], [], [], []  # time, node, node, weight
    for k1, v1 in time_edges.items():
        for k2, v2 in v1.items():  # k2 = edge = (u,v)
            if k2[0] != k2[1]:
                l1.append(k1)
                l2.append(k2[0])
                l3.append(k2[1])
                l4.append(v2)
    data_graph = pd.DataFrame(
        list(zip(l1, l2, l3, l4)), columns=['t', 'u', 'v', 'w']
    )

    # Scale edge weights to range [0-1]
    X = [[entry] for entry in data_graph['w']]
    if save_weights: np.savetxt(file_out[8], X, delimiter=',', fmt='%s')

    # Max-Min Normalizer (produce many zeros)
    # transformer = MinMaxScaler()
    # X_scaled = transformer.fit_transform(X)
    # Returning column to row vector again
    # X_scaled = [entry[0] for entry in X_scaled]

    # Quantile normalizer (normal distribution)
    # transformer = QuantileTransformer()

    # Quantile normalizer (uniform distribution)
    transformer = QuantileTransformer(
        n_quantiles=1000,
        output_distribution='uniform',
    )
    X_scaled = transformer.fit_transform(X)
    X_scaled = [entry[0] for entry in X_scaled]

    # Normalize by dividing to max
    # X_max = max(data_graph['w'])
    # X_scaled = [entry[0] / X_max for entry in X]

    # Fixing 0's and 1's entries
    # X_scaled = [entry if entry != 1 else 0.99 for entry in X_scaled]
    # X_scaled = [entry if entry > 0 else 0.1 for entry in X_scaled]

    # Scale everything between [a,b] or [0.5,1]
    # Because we do not want these weight become less than temporal weights
    # X_scaled = (b - a) * ((X_scaled - min(X_scaled)) / (max(X_scaled) - min(X_scaled))) + a
    X_scaled = (0.5 * np.array(X_scaled)) + 0.5

    # Rounding to X decimal point
    # X_scaled = [round(entry, 2) for entry in X_scaled]  # List
    X_scaled = np.round(X_scaled, 2)  # Array

    # Save back scaled weights to DF
    data_graph['w'] = X_scaled

    # Save scaled weights to file
    if save_weights:
        np.savetxt(file_out[10], X_scaled, delimiter=',', fmt='%s')

    # Save network to DB
    if save_network_db:
        data_graph[['u', 'v', 't', 'w']].to_sql(
            name='bluetooth_edgelist',
            con=sqlite3.connect(path2),
            if_exists='replace',
            index_label='id'
        )

    # Save network to file
    if save_network_csv:
        data_graph[['u', 'v', 't',
                    'w']].to_csv(file_out[2], header=False, index=False)

    # Add edges to network object
    for row in data_graph.itertuples(index=True, name='Pandas'):
        graph.add_edge(
            getattr(row, 'u'),
            getattr(row, 'v'),
            t=getattr(row, 't'),
            w=getattr(row, 'w')
        )

    # Save graph to file as netX object
    if save_network_file:
        nx.write_gpickle(graph, file_out[0])

    # Save timestamps
    if save_times:
        times_set = set()
        for u, v, w in graph.edges(data=True):
            times_set.add(w['t'])
        times = pd.Series(sorted(list(times_set)))
        np.savetxt(file_out[4], times, delimiter=',', fmt='%s')

    # Save nodes
    if save_nodes:
        # List of nodes in the graph
        nodes = pd.Series(sorted(list(graph.nodes)))
        # Save node list in a file "node.csv"
        np.savetxt(file_out[6], nodes, delimiter=',', fmt='%s')

    # Bipartite network edge data
    l1, l2, l3, l4 = [], [], [], []
    for k1, v1 in time_bipartite_edges.items():
        for k2, v2 in v1.items():  # k2 = edge = (u,v)
            if k2[0] != k2[1]:
                l1.append(k1)
                l2.append(k2[0])
                l3.append(k2[1])
                l4.append(v2)
    data_bi_graph = pd.DataFrame(
        list(zip(l1, l2, l3, l4)), columns=['t', 'u', 'v', 'w']
    )

    # Weights
    X = [[entry] for entry in data_bi_graph['w']]
    if save_weights: np.savetxt(file_out[9], X, delimiter=',', fmt='%s')
    transformer = QuantileTransformer(
        n_quantiles=100,
        output_distribution='uniform',
    )
    X_scaled = transformer.fit_transform(X)
    X_scaled = [entry[0] for entry in X_scaled]
    if save_weights:
        np.savetxt(file_out[11], X_scaled, delimiter=',', fmt='%s')
    data_bi_graph['w'] = X_scaled

    # Save bipartite to DB
    if save_network_db:
        data_bi_graph[['u', 'v', 't', 'w']].to_sql(
            name='bluetooth_bipartite_edgelist',
            con=sqlite3.connect(path2),
            if_exists='replace',
            index_label='id'
        )

    # Save bipartite to file
    if save_network_csv:
        data_bi_graph[['u', 'v', 't',
                       'w']].to_csv(file_out[3], header=False, index=False)

    # Add nodes and edges to bipartite network oject
    # We need to add a prefix "u_" for users & "b_" for BT devices to the node id
    # So that we can distinguish them from each others
    for row in data_bi_graph.itertuples(index=True, name='Pandas'):
        # In bluetooth connections, user devices ID are repeated in all BT devices
        # So there is no need to differentiate between them
        bipartite.add_edge(
            getattr(row, 'u'),
            getattr(row, 'v'),
            t=getattr(row, 't'),
            w=getattr(row, 'w')
        )

    # Save graph
    if save_network_file:
        nx.write_gpickle(bipartite, file_out[1])

    # Save timestamps
    if save_times:
        times_set = set()
        for u, v, w in bipartite.edges(data=True):
            times_set.add(w['t'])
        times_bipartite = pd.Series(sorted(list(times_set)))
        np.savetxt(file_out[5], times_bipartite, delimiter=',', fmt='%s')

    # Save nodes
    if save_nodes:
        # List of nodes in the bipartite graph
        nodes = pd.Series(sorted(list(bipartite.nodes)))
        # Save node list in a file "node.csv"
        np.savetxt(file_out[7], nodes, delimiter=',', fmt='%s')
        # pd.DataFrame(sorted(list(times))).to_csv(file_in[2], header=None, index=False)

    # Print network statistics
    if output_network:
        print(nx.info(graph))
        print(f'Number of times = {len(times)}')

    return graph


def temporal_bt_read(
    folder_in=NETWORK,
    file_in=['bt_temporal_network.gpickle'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    Reads temporal graph
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)

    # Network
    graph = nx.MultiDiGraph()

    # Read from file
    if os.path.exists(file_in[0]):
        graph = nx.read_gpickle(file_in[0])
    else:
        print('File not found')
        return None

    # Print graph statistics
    if output: print(nx.info(graph))

    return graph


def temporal_bt_times_read(
    folder_in=NETWORK,
    file_in=['bt_temporal_times.csv'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    Reads timestamps of temporal graph
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)

    # Times
    times = []

    # Read from file
    if os.path.exists(file_in[0]):
        times = pd.read_csv(
            file_in[0], index_col=False, header=None, names=['times']
        ).iloc[:, 0]
        # Change type (str -> datetime)
        times = times.astype('datetime64[ns]')
    else:
        print('File not found')
        return None

    return times


def temporal_bt_nodes_read(
    folder_in=NETWORK,
    file_in=['bt_temporal_nodes.csv'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    Reads nodes of temporal graph
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)

    # Nodes
    nodes = []

    # Read from file
    if os.path.exists(file_in[0]):
        nodes = pd.read_csv(
            file_in[0], index_col=False, header=None, names=['nodes']
        ).iloc[:, 0]
    else:
        print('File not found')
        return None

    return nodes


# --------------
# Static Network
# --------------


def static_bt(
    folder_in=NETWORK,
    folder_out=NETWORK,
    file_in=['bt_temporal_network.gpickle'],
    file_out=['bt_static_network.gpickle', 'bt_static_edgelist.csv'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    temporal=None,
    tmies=None,
    undirected=True,
    output_network=True,
    save_network_csv=True,
    save_network_file=True,
):
    """
    Convert input temporal network to the (aggregated) static network
    the edge weights are sum of temporal interactions over entire time window
    """
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Read temporal network from file
    if temporal is None:
        temporal = temporal_bt_read(
            folder_in,
            file_in,
            label_folder_in,
            label_file_in,
        )

    # Create static graph
    graph = nx.Graph()
    graph.name = 'Static Network'

    # Update static network edges
    for u, v, data in temporal.edges(data=True):
        t = 1 if 't' in data else 0
        if graph.has_edge(u, v):
            graph[u][v]['s'] += t
            graph[u][v]['w'] = graph[u][v]['w'] + data['w']
        else:
            graph.add_edge(u, v, s=t, w=data['w'])

    # Fix edge weights, if network is directed, because they have been counted twice
    if undirected:
        for u, v, data in graph.edges(data=True):
            graph[u][v]['s'] //= 2
            graph[u][v]['w'] /= 2
            # Mean of weight
            graph[u][v]['w'] /= graph[u][v]['s']

    # Save the network
    if save_network_file:
        nx.write_gpickle(graph, file_out[0])

    if save_network_csv:
        # (1)
        # nx.write_edgelist(graph, file_out[1], data=True)
        nx.write_weighted_edgelist(graph, file_out[1], delimiter=',')

    # Densitiy
    den = nx.classes.function.density(graph)
    # Is network connected
    con = nx.algorithms.components.is_connected(graph)
    # Connected components
    cc = 1
    # Diameter
    dim = graph.number_of_nodes()
    if not con:
        cc = nx.algorithms.components.number_connected_components(graph)
        largest_cc = max(nx.connected_components(graph), key=len)
        dim = nx.algorithms.distance_measures.diameter(largest_cc)
    else:
        dim = nx.algorithms.distance_measures.diameter(graph)

    # Print network statistics
    if output_network:
        print(nx.info(graph))
        if con:
            print('Network is connected.')
        else:
            print('Network is not connected.')
        print('Density =', den)
        print('Number of connected components =', cc)
        print('Diameter =', dim)

    return graph


def static_bt_read(
    folder_in=NETWORK,
    file_in=['bt_static_network.gpickle'],
    label_folder_in='',
    label_file_in='',
    output=False,
    stat=False,
):
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)

    # Static graph
    graph = nx.Graph()

    # Read the network from file
    if os.path.exists(file_in[0]):
        graph = nx.read_gpickle(file_in[0])

    # Network statistics
    if stat:
        # Densitiy
        den = nx.classes.function.density(graph)
        # Is network connected
        con = nx.algorithms.components.is_connected(graph)
        # Connected components
        cc = 1
        # Diameter
        dim = graph.number_of_nodes()
        if not con:
            cc = nx.algorithms.components.number_connected_components(graph)
            largest_cc = max(nx.connected_components(graph), key=len)
            dim = nx.algorithms.distance_measures.diameter(largest_cc)
        else:
            dim = nx.algorithms.distance_measures.diameter(graph)

        # Print network statistics
        if output:
            print(nx.info(graph))
            if con:
                print('Network is connected.')
            else:
                print('Network is not connected.')
            print('Density =', den)
            print('Number of connected components =', cc)
            print('Diameter =', dim)

    return graph


# --------------------
# Time-Ordered Network
# --------------------


def ton_bt(
    folder_in=[DB, NETWORK],
    folder_out=NETWORK,
    file_in=[
        'uiuc.db',
        'bt_temporal_network.gpickle',
        'bt_temporal_times.csv',
    ],
    file_out=['bt_ton_network.gpickle', 'bt_ton_edgelist.csv'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    temporal=None,
    times=None,
    directed=True,
    teleport=False,
    loop=False,
    trans=True,
    output_network=True,
    save_network_db=False,
    save_network_csv=True,
    save_network_file=True
):
    """
    Create a (directed) time-ordered (temporal) network
    the LIGHT version do not set any edge weight attribute, keeping the model light
    
    Parameters
    ----------
    directed : bool
        add bi-directional temporal edges i.e. t <-> t+1
    teleport :bool
        add temporal teleportation edges
    loop : bool
        connect nodes at last timestamp to first i.e. T -> t0
    """
    # Edit paths
    path1 = path_edit(
        [file_in[0]],
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Read temporal networks and timestamps
    if temporal is None:
        temporal = temporal_bt_read(
            folder_in[1],
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )
    if times is None:
        times = temporal_bt_times_read(
            folder_in[1],
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )

    # TON graph
    graph = nx.DiGraph()
    graph.name = 'Time-Ordered Network'

    # Size of timestamp list
    T = len(times)

    # Time -> Index
    time_index = dict((v, i) for i, v in enumerate(times))

    # Size of nodes
    N = temporal.number_of_nodes()

    # Node -> Index
    nodes = pd.Series(sorted(list(temporal.nodes)))
    node_index = dict((v, i) for i, v in enumerate(nodes))

    # Size of edges
    L = temporal.number_of_edges()

    # Node (index) -> horizontal edges
    node_edges = {}
    for n in range(N):
        node_edges[n] = [(N * t + n, N * (t + 1) + n) for t in range(T)]

    # Horizontal edges
    if directed:
        for node, edges in node_edges.items():
            for i in range(len(edges)):  # [0,T]
                # Add edges: node(i) -> node(i+1)
                graph.add_edge(edges[i][0], edges[i][1])
                # With temporal teleportation
                if teleport:
                    for j in range(i + 1, len(edges)):
                        graph.add_edge(edges[i][0], edges[j][1])
        # With temporal loop
        if loop:
            for node in node_edges:
                graph.add_edge(node_edges[node][-1][1], node_edges[node][0][0])
    else:  # Undirected
        for node, edges in node_edges.items():
            for i in range(len(edges)):
                graph.add_edge(edges[i][0], edges[i][1])
                # Backward horizontal edge (i.e. moving back in time)
                graph.add_edge(edges[i][1], edges[i][0])
                if teleport:
                    for j in range(i + 1, len(edges)):
                        graph.add_edge(edges[i][0], edges[j][1])
                        # Backward teleportation in time
                        graph.add_edge(edges[j][1], edges[i][0])
        # With tempora loop
        if loop:
            for node in node_edges:
                graph.add_edge(node_edges[node][-1][1], node_edges[node][0][0])
                graph.add_edge(node_edges[node][0][0], node_edges[node][-1][1])

    # Crossed edges
    if directed:  # Directed
        for u, v, edge_data in temporal.edges(data=True):
            u_index = node_index[u]
            v_index = node_index[v]
            t_index = time_index[edge_data['t']]
            graph.add_edge(
                u_index + t_index * N,
                v_index + (t_index + 1) * N,
                w=edge_data['w']  # Only edge weight is set in light version
            )
    else:  # Undirected
        for u, v, edge_data in temporal.edges(data=True):
            u_index = node_index[u]
            v_index = node_index[v]
            t_index = time_index[edge_data['t']]
            graph.add_edge(
                u_index + t_index * N,
                v_index + (t_index + 1) * N,
                w=edge_data['w']
            )
            graph.add_edge(
                v_index + (t_index + 1) * N,
                u_index + t_index * N,
                w=edge_data['w']
            )

    # Transitive closure
    trans_num = 0
    if trans:
        for t in range(T):
            snap_nodes = [(t * N) + n for n in range(N)]
            snap_nodes.extend([((t + 1) * N) + n for n in range(N)])
            snap_graph = graph.subgraph(snap_nodes)
            A = nx.to_numpy_matrix(snap_graph)
            A_t = A[:len(A) // 2, len(A) // 2:]
            snap_trans = nx.to_numpy_matrix(
                nx.transitive_closure(
                    nx.from_numpy_matrix(A_t, create_using=nx.DiGraph)
                )
            )
            # Compare edges of transitive closure with edges we had before
            # Find new edges, add them to network
            snap_edges = np.transpose(np.nonzero(A_t != snap_trans))
            snap_weights = np.tile(
                0.5 * np.random.sample(len(snap_edges) // 2) + 0.5, 2
            )
            # index of new edges should be converted into node ID in network
            for r in range(len(snap_edges)):
                if not graph.has_edge(
                    snap_nodes[snap_edges[r][0]],
                    snap_nodes[snap_edges[r][1] + N]
                ):
                    trans_num += 1  # Counter of transitive edges
                    graph.add_edge(
                        snap_nodes[snap_edges[r][0]],
                        snap_nodes[snap_edges[r][1] + N],
                        w=snap_weights[r],
                        trans=True
                    )
                    if not directed:
                        graph.add_edge(
                            snap_nodes[snap_edges[r][0]] + N,
                            snap_nodes[snap_edges[r][1]],
                            w=snap_weights[r],
                            trans=True
                        )

    # Save network to file
    if save_network_file:
        nx.write_gpickle(graph, file_out[0])

    # Save network edgelist
    if save_network_csv:
        nx.write_weighted_edgelist(graph, file_out[1], delimiter=',')

    # Save network to database
    if save_network_db:
        edge_list = pd.DataFrame.from_dict(graph.edges)
        edge_list.columns = ['u', 'v']
        edge_list.to_sql(
            name='bluetooth_time_ordered_edgelist',
            con=sqlite3.connect(path1),
            if_exists='replace',
            index_label='id'
        )

    # Print network statics
    if output_network:
        print(nx.info(graph))
        if trans:
            print(f'Number of transitive edges = {trans_num}')

    return graph


def ton_bt_full(
    folder_in=[DB, NETWORK],
    folder_out=NETWORK,
    file_in=[
        'uiuc.db',
        'bt_temporal_network.gpickle',
        'bt_temporal_times.csv',
    ],
    file_out=[
        'bt_ton_network.gpickle',
        'bt_ton_edgelist.csv',
        'bt_ton_delta.csv',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    temporal=None,
    times=None,
    directed=True,
    trans=True,
    teleport=False,
    loop=False,
    output_delta=True,
    output_network=True,
    save_delta=True,
    save_network_csv=True,
    save_network_file=True
):
    """
    Create a (directed) time-ordered (temporal) network
    the FULL version set a nubmer of edge weight attributes as position and color
    
    Parameters
    ----------
    directed : bool
        add bi-directional temporal edges i.e. t <-> t+1
    teleport :bool
        add temporal teleportation edges
    loop : bool
        connect nodes at last timestamp to first i.e. T -> t0
    """
    # Edit paths
    path1 = path_edit(
        [file_in[0]],
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Read temporal networks and timestamps
    if temporal is None:
        temporal = temporal_bt_read(
            folder_in[1],
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )
    if times is None:
        times = temporal_bt_times_read(
            folder_in[1],
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )

    # TON graph
    graph = nx.DiGraph()
    graph.name = 'Full Time-Ordered Network'

    # Size of timestamp list
    T = len(times)

    # Time -> Index
    time_index = dict((v, i) for i, v in enumerate(times))

    # Size of nodes
    N = temporal.number_of_nodes()

    # Node -> Index
    nodes = pd.Series(sorted(list(temporal.nodes)))
    node_index = dict((v, i) for i, v in enumerate(nodes))

    # Size of edges
    L = temporal.number_of_edges()

    # Node (index) -> horizontal edges
    node_edges = {}
    for n in range(N):
        node_edges[n] = [(N * t + n, N * (t + 1) + n) for t in range(T)]

    # Colors for nodes at differnt timestamp
    colors = []
    cmap = cm.get_cmap('Wistia', T + 1)
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        colors.append(matplotlib.colors.rgb2hex(rgb))

    # Time delta or time difference between all consecutive timestamps
    # First convert delta to second and then hour
    # Add '1' to the begginig so that len(delta) == len(times)
    delta = []
    times = list(times)
    delta_temp = pd.Series(pd.Series(times[1:]) - pd.Series(times[:T - 1]))
    delta = [int(ts.total_seconds() // 3600) for ts in delta_temp]
    delta.insert(0, 1)  # (postion, value)

    # Change time and delta to series
    times = pd.Series(times)
    delta = pd.Series(delta)

    if save_delta:  # save delta
        np.savetxt(file_out[2], delta.values, delimiter=',', fmt='%s')

    if output_delta:
        # delta_count = pd.Series(Counter(delta)).sort_index()
        delta_count = delta.value_counts(normalize=True)
        print("Delta Distribution\n------------------")
        print(delta_count)

    # Horizontal edges
    for node, edges in node_edges.items():
        # Add the first node at the first timestamp
        graph.add_node(node, c=colors[0], p=(0, node))
        for i in range(len(edges)):  # i = time, and in range [0,T]
            # Add the edge (u,v)
            graph.add_edge(
                edges[i][0],
                edges[i][1],
                t=times[i],  # Timestamp
                d=delta[i],  # Delta or temporal distance
                c='silver'  # Color
            )
            # Backward horizontal edge (i.e. moving back in time)
            if not directed:
                graph.add_edge(
                    edges[i][1],
                    edges[i][0],
                    t=times[i],
                    d=delta[i],
                    c='silver'
                )
            # Then set attribute of second node of just created edge (u,v)
            graph.nodes[edges[i][1]]['c'] = colors[i + 1]
            graph.nodes[edges[i][1]]['p'] = (i + 1, node)
            # Temporal teleportation
            if teleport:
                for j in range(i + 1, len(edges)):
                    graph.add_edge(
                        edges[i][0], edges[j][1], d=sum(delta[i:j]), c='gold'
                    )
                    if not directed:
                        graph.add_edge(
                            edges[j][0],
                            edges[i][1],
                            d=sum(delta[i:j]),
                            c='gold'
                        )
    # Temporal loop
    if loop:
        for node in node_edges:
            graph.add_edge(
                node_edges[node][-1][1],
                node_edges[node][0][0],
                d=sum(delta),
                c='orange'
            )
            if not directed:
                graph.add_edge(
                    node_edges[node][0][0],
                    node_edges[node][-1][1],
                    d=sum(delta),
                    c='orange'
                )

    # Crossed edges
    for u, v, edge_data in temporal.edges(data=True):
        u_index = node_index[u]
        v_index = node_index[v]
        t_index = time_index[edge_data['t']]
        graph.add_edge(
            u_index + t_index * N,
            v_index + (t_index + 1) * N,
            w=edge_data['w'],
            t=edge_data['t'],
            d=delta[t_index],
            c='black'
        )
        if not directed:
            graph.add_edge(
                v_index + (t_index + 1) * N,
                u_index + t_index * N,
                w=edge_data['w'],
                t=edge_data['t'],
                d=delta[t_index],
                c='black'
            )

    # Transitive closure
    trans_num = 0
    if trans:
        for t in range(T):
            snap_nodes = [(t * N) + n for n in range(N)]
            snap_nodes.extend([((t + 1) * N) + n for n in range(N)])
            snap_graph = graph.subgraph(snap_nodes)
            A = nx.to_numpy_matrix(snap_graph)
            A_t = A[:len(A) // 2, len(A) // 2:]
            snap_trans = nx.to_numpy_matrix(
                nx.transitive_closure(
                    nx.from_numpy_matrix(A_t, create_using=nx.DiGraph)
                )
            )
            # Compare edges of transitive closure with edges we had before
            # Find new edges, add them to network
            snap_edges = np.transpose(np.nonzero(A_t != snap_trans))
            snap_weights = np.tile(
                0.5 * np.random.sample(len(snap_edges) // 2) + 0.5, 2
            )
            # index of new edges should be converted into node ID in network
            for r in range(len(snap_edges)):
                if not graph.has_edge(
                    snap_nodes[snap_edges[r][0]],
                    snap_nodes[snap_edges[r][1] + N]
                ):
                    trans_num += 1  # Counter of transitive edges
                    graph.add_edge(
                        snap_nodes[snap_edges[r][0]],
                        snap_nodes[snap_edges[r][1] + N],
                        w=snap_weights[r],
                        t=times[t],
                        d=delta[t],
                        trans=True,
                        c='red'
                    )  # Only for trans edges
                    if not directed:
                        graph.add_edge(
                            snap_nodes[snap_edges[r][0]] + N,
                            snap_nodes[snap_edges[r][1]],
                            w=snap_weights[r],
                            t=times[t],
                            d=delta[t],
                            trans=True,
                            c='red'
                        )

    # Save network to file
    if save_network_file:
        nx.write_gpickle(graph, file_out[0])

    # Save network edgelist
    if save_network_csv:
        nx.write_weighted_edgelist(graph, file_out[1], delimiter=',')

    # Print network statics
    if output_network:
        print(nx.info(graph))
        if trans:
            print(f'Number of transitive edges = {trans_num}')

    return graph


def ton_bt_to_temporal(
    folder_in=NETWORK,
    folder_out=NETWORK,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
    ],
    file_out=[
        'bt_temporal_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    label_graph='',
    temporal=None,
    trans_remove=True,
    output_network=True,
    save_times=True,
    save_nodes=True,
    save_network_file=True
):
    """
    Convert (directed) time-ordered network (TON) to temporal network
    """
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Create empty network
    graph = nx.MultiDiGraph()
    graph_name = 'Temporal Network'
    if len(label_graph) > 0: graph_name = graph_name + ' ' + label_graph
    graph.name = graph_name

    # Read temporal networks from file
    if temporal is None:
        temporal = temporal_bt_read(
            folder_in,
            [file_in[0]],
            label_folder_in,
            label_file_in,
        )

    # Nodes
    nodes = temporal_bt_nodes_read(
        folder_in,
        [file_in[1]],
        label_folder_in,
        label_file_in,
    )
    # N = file_line_count(path_edit(folder_in,[file_in[1]],label_folder_in,label_file_in,)[0])
    N = len(nodes)

    # Timestamp
    times = temporal_bt_times_read(
        folder_in,
        [file_in[2]],
        label_folder_in,
        label_file_in,
    )
    # T = file_line_count(path_edit(folder_in,[file_in[2]],label_folder_in,label_file_in,)[0])
    T = len(times)

    # Calculate current timestamp list from graph
    # In case, in node removal caused loosing some timestamp comparing to original
    times_set = set()

    # Iterate edges and add crossed ones back to temporal network object
    for u, v, data in temporal.edges(data=True):
        parent_u = u % N
        parent_v = v % N
        time_uv = u // N  # OR v // N - 1
        time_delta = abs(v - u) // N
        # Crossed edge
        if parent_u != parent_v:  # and time_delta == 1:
            if trans_remove and data.get('trans', False):
                # If the the edge is transitive and we want to ignore trans -> skip
                continue
            graph.add_edge(
                parent_u,
                parent_v,
                t=times.loc[time_uv],
                w=data['w'],
            )
            # Save timestamp to the new time set
            times_set.add(times.loc[time_uv])

    # Convert times set to series and save
    times_new = pd.Series(sorted(list(times_set)))
    nodes_new = pd.Series(sorted(list(graph.nodes)))

    # Save graph
    if save_network_file: nx.write_gpickle(graph, file_out[0])

    # Save nodes
    if save_nodes:
        np.savetxt(file_out[1], nodes_new, delimiter=',', fmt='%s')

    # Save times
    if save_times:
        np.savetxt(file_out[2], times_new, delimiter=',', fmt='%s')

    # Print network statistics
    if output_network:
        print(nx.info(graph))
        print(f'Number of times: {len(times_new)}')

    return graph


def ton_bt_read(
    folder_in=NETWORK,
    file_in=['bt_ton_network.gpickle'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    Reads time-ordered network (TON) graph of Bluetooth connections
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)
    graph = nx.read_gpickle(file_in[0])
    if output: print(nx.info(graph))
    return graph


def ton_bt_full_read(
    folder_in=NETWORK,
    file_in=['bt_ton_full_network.gpickle'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    bt_ton_full_network.gpickle
    Reads full version of time-ordered network (TON) graph of Bluetooth connections
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)
    graph = nx.read_gpickle(file_in[0])
    if output: print(nx.info(graph))
    return graph


# ------------
# Reachability
# ------------


# def reachability(
#     folder_in=NETWORK,
#     folder_out=NETWORK,
#     file_in=[
#         'bt_ton_network.gpickle',
#         'bt_temporal_nodes.csv',
#         'bt_temporal_times.csv',
#         'bt_ton_weights.csv',
#     ],
#     file_out=[
#         'reachability.csv',
#         'reachability.p',
#     ],
#     label_folder_in='',
#     label_folder_out='',
#     label_file_in='',
#     label_file_out='',
#     graph=None,
#     times=None,
#     nodes=None,
#     ew=None,
#     output=True,
#     save=True,
# ):
#     # Edit paths
#     file_out = path_edit(
#         file_out,
#         folder_out,
#         label_file_out,
#         label_folder_out,
#     )

#     # Reading graph
#     if graph is None:
#         graph = ton_bt_read(
#             folder_in,
#             [file_in[0]],
#             label_folder_in,
#             label_file_in,
#         )

#     # Reading nodes
#     if nodes is None:
#         nodes = temporal_bt_nodes_read(
#             folder_in,
#             [file_in[1]],
#             label_folder_in,
#             label_file_in,
#         )
#     N = len(nodes)

#     # Reading times
#     if times is None:
#         times = temporal_bt_times_read(
#             folder_in,
#             [file_in[2]],
#             label_folder_in,
#             label_file_in,
#         )
#     T = len(times)

#     # All possible paths
#     all_paths = {}
#     for n1 in range(N):
#         for t1 in range(T + 1):
#             for n2 in range(N):
#                 for t2 in range(T + 1):
#                     if n1 != n2 and t1 < t2:
#                         all_paths[(n1, n2, t1, t2)] = {'d': np.inf}

#     # All existing network shortest path lenght (SPL)
#     spl = dict(nx.all_pairs_shortest_path_length(graph))

#     # Convert N1 -> N2 to Path of (Parent N1, Parent N2, t1, t2)
#     # Save all reachable paths P(n1,n2,t1,t2) in a dataframe
#     reached = []
#     r = 0

#     for n1 in spl:
#         # Parent of source node
#         p1 = n1 % N
#         t1 = n1 // N
#         for n2 in spl[n1]:
#             p2 = n2 % N
#             t2 = n2 // N
#             # Nodes with different parents and timestamps
#             if p1 != p2:
#                 # Add path lenght
#                 reached.append((p1, p2, t1, t2, spl[n1][n2]))
#                 # Add temporal distance to path
#                 all_paths[(p1, p2, t1, t2)]['d'] = spl[n1][n2]
#                 # Add the number of reached paths
#                 r += 1

#     if output:
#         print(f'Number of reachable paths is {r}', end='')
#         print(f'from a total of {len(all_paths)} possoble paths or ', end='')
#         print(f'{r/len(all_paths)*100:0.2f} %')

#     # Create dataframe of reachability -> Easy to use for path filtering
#     df = pd.DataFrame(reached, columns=['n1', 'n2', 't1', 't2', 'd'])

#     # Check paths for hop-number or node-distance
#     cc = 0  # Counter
#     cr = 0  # Counter % passed
#     cl = int(0.01 * r)  # Counter limit = 1 % of all paths
#     for (p1, p2, t1, t2), val in all_paths.items():
#         d = val['d']
#         if d < np.inf:
#             cc += 1
#             if cc % cl == 0:
#                 cr += 1
#                 print(f'Processed {cr} % of all data ...')
#             mind = 0
#             # Check if there is a path with shorter temporal distance
#             q = df.loc[(df['n1'] == p1) & (df['n2'] == p2) & (df['t1'] >= t1) &
#                        (df['t2'] <= t2) & (df['d'] < d)]
#             # Check if it is direct edge or 1-hop between two nodes
#             # But in a shoeter time interval included in larger time window
#             if q.d.min() == 1:
#                 # ndistances.append(1)
#                 mind = 1
#                 all_paths[(p1, p2, t1, t2)]['h'] = 1

#             # If it is not a direct connection
#             # And there are some intermediate nodes incolved
#             if q.d.min() > 1:
#                 # Look at all shortest paths in that time range
#                 sps = [
#                     np.array(p) % N for p in nx.all_shortest_paths(
#                         graph,
#                         source=(N * t1) + p1,
#                         target=(N * t2) + p2,
#                     )
#                 ]
#                 # Extract  intermediate nodes
#                 # And save in a dictionary based on path length
#                 spd = defaultdict(list)
#                 for x in sps:
#                     # Remove consecutive nodes (same node, next or previous time)
#                     temp = [
#                         v for i, v in enumerate(x) if i == 0 or v != x[i - 1]
#                     ]
#                     spd[len(temp)].append(temp)
#                 # Check only the shortest extracted paths
#                 mind = min(spd.keys())
#                 spd = spd[mind]
#                 # Minus 1 (-1) because hop length is 1 less than number of nodes
#                 all_paths[(p1, p2, t1, t2)]['h'] = mind - 1
#                 # Unique intermediate nodes invole in shortest path
#                 paths = paths = set()
#                 for x in spd:
#                     paths.add(tuple(x[1:-1]))
#                 all_paths[(p1, p2, t1, t2)]['p'] = [list(x) for x in paths]
#         else:
#             # Path does not exist
#             # Distance is infinity
#             # Then add to REACHED list to create new DF
#             reached.append((p1, p2, t1, t2, np.inf))

#     # Create reachability again LOL
#     df = pd.DataFrame(reached, columns=['n1', 'n2', 't1', 't2', 'd'])
#     # Then sort ...
#     df.sort_values(
#         by=['n1', 'n2', 't1', 't2'],
#         inplace=True,
#         ignore_index=True,
#     )
#     # Finally save reachability dataframe and Hop dictionary
#     if save:
#         df.to_csv(file_out[0], header=False, index=False)
#         with open(file_out[1], 'wb') as fp:
#             pickle.dump(all_paths, fp, protocol=pickle.HIGHEST_PROTOCOL)


# def reachability_create(
#     folder_in=NETWORK,
#     folder_out=NETWORK,
#     file_in=[
#         'bt_ton_network.gpickle',
#         'bt_temporal_nodes.csv',
#         'bt_temporal_times.csv',
#         'bt_ton_weights.csv',
#     ],
#     file_out=[
#         'paths_reached.csv',
#         'paths_all.csv',
#         'paths_all.p',
#     ],
#     label_folder_in='',
#     label_folder_out='',
#     label_file_in='',
#     label_file_out='',
#     graph=None,
#     times=None,
#     nodes=None,
#     ew=None,
#     output=True,
#     save=True,
# ):
#     # Edit paths
#     file_out = path_edit(
#         file_out,
#         folder_out,
#         label_file_out,
#         label_folder_out,
#     )

#     # Reading graph
#     if graph is None:
#         graph = ton_bt_read(
#             folder_in,
#             [file_in[0]],
#             label_folder_in,
#             label_file_in,
#         )

#     # Reading nodes
#     if nodes is None:
#         nodes = temporal_bt_nodes_read(
#             folder_in,
#             [file_in[1]],
#             label_folder_in,
#             label_file_in,
#         )
#     N = len(nodes)

#     # Reading times
#     if times is None:
#         times = temporal_bt_times_read(
#             folder_in,
#             [file_in[2]],
#             label_folder_in,
#             label_file_in,
#         )
#     T = len(times)

#     # Reading edge weight
#     # if ew is None:
#     #     ew = ew_read(
#     #         folder_in,
#     #         [file_in[3]],
#     #         label_folder_in,
#     #         label_file_in,
#     #     )

#     # All possible paths
#     all_paths = {}
#     for n1 in range(N):
#         for t1 in range(T + 1):
#             for n2 in range(N):
#                 for t2 in range(T + 1):
#                     if n1 != n2 and t1 < t2:
#                         all_paths[(n1, n2, t1, t2)] = {'d': np.inf}

#     # All existing network shortest path lenght (SPL)
#     spl = dict(nx.all_pairs_shortest_path_length(graph))

#     # Convert N1 -> N2 to Path of (Parent N1, Parent N2, t1, t2)
#     # Save all reachable paths P(n1,n2,t1,t2) in a dataframe
#     reached = []
#     r = 0

#     for n1 in spl:
#         # Parent of source node
#         p1 = n1 % N
#         t1 = n1 // N
#         for n2 in spl[n1]:
#             p2 = n2 % N
#             t2 = n2 // N
#             # Nodes with different parents and timestamps
#             if p1 != p2:
#                 # Add path lenght
#                 reached.append((p1, p2, t1, t2, spl[n1][n2]))
#                 # Add temporal distance to path
#                 all_paths[(p1, p2, t1, t2)]['d'] = spl[n1][n2]
#                 # Add the number of reached paths
#                 r += 1

#     if output:
#         print(f'Number of reachable paths is {r}', end='')
#         print(f'from a total of {len(all_paths)} possoble paths or ', end='')
#         print(f'{r/len(all_paths)*100:0.2f} %')

#     # Create dataframe of reachability -> Easy to use for path filtering
#     df = pd.DataFrame(reached, columns=['n1', 'n2', 't1', 't2', 'd'])
#     reached.clear()
#     # Then sort ...
#     df.sort_values(
#         by=['n1', 'n2', 't1', 't2'],
#         inplace=True,
#         ignore_index=True,
#     )
#     # Finally save reachability dataframe and Hop dictionary
#     if save:
#         df.to_csv(file_out[0], header=False, index=False)

#     # Check paths for hop-number or node-distance
#     cc = 0  # Counter
#     cr = 0  # Counter % passed
#     cl = int(0.01 * r)  # Counter limit = 1 % of all paths
#     reached2 = []
#     for (p1, p2, t1, t2), val in all_paths.items():
#         d = val['d']
#         if d < np.inf:
#             cc += 1
#             if cc % cl == 0:
#                 cr += 1
#                 print(f'Processed {cr} % of all data ...')
#             # Check if there is a path with shorter temporal distance
#             q = df.loc[(df['n1'] == p1) & (df['n2'] == p2) & (df['t1'] >= t1) &
#                        (df['t2'] <= t2)]
#             mind = q.d.min()
#             # This is minimum temporal distance
#             all_paths[(p1, p2, t1, t2)]['h'] = mind
#             reached2.append[(p1, p2, t1, t2, d, mind)]
#         else:
#             # Path does not exist
#             # Distance is infinity
#             # Then add to REACHED list to create new DF
#             all_paths[(p1, p2, t1, t2)]['h'] = np.inf
#             reached2.append((p1, p2, t1, t2, np.inf, np.inf))

#     # Create reachability again LOL
#     df = pd.DataFrame(reached2, columns=['n1', 'n2', 't1', 't2', 'd', 'h'])
#     reached2.clear()
#     # Then sort ...
#     df.sort_values(
#         by=['n1', 'n2', 't1', 't2'],
#         inplace=True,
#         ignore_index=True,
#     )
#     # Finally save reachability dataframe and Hop dictionary
#     if save:
#         df.to_csv(file_out[1], header=False, index=False)
#         with open(file_out[2], 'wb') as fp:
#             pickle.dump(all_paths, fp, protocol=pickle.HIGHEST_PROTOCOL)


# def reachability_calculate(
#     folder_in=NETWORK,
#     folder_out=NETWORK,
#     file_in=[
#         'bt_ton_network.gpickle',
#         'bt_temporal_nodes.csv',
#         'bt_temporal_times.csv',
#         'bt_ton_weights.csv',
#     ],
#     file_out=[
#         'paths_all.p',
#         'paths_all.csv',
#     ],
#     label_folder_in='',
#     label_folder_out='',
#     label_file_in='',
#     label_file_out='',
#     graph=None,
#     times=None,
#     nodes=None,
#     ew=None,
#     percentage=1,
#     output=True,
#     save=True,
# ):
#     """
#     Every time this function only work on 1% of the data from 1 .. 100
#     """
#     # Edit paths
#     file_out = path_edit(
#         file_out,
#         folder_out,
#         label_file_out,
#         label_folder_out,
#     )

#     # Reading graph
#     if graph is None:
#         graph = ton_bt_read(
#             folder_in,
#             [file_in[0]],
#             label_folder_in,
#             label_file_in,
#         )

#     # Reading nodes
#     if nodes is None:
#         nodes = temporal_bt_nodes_read(
#             folder_in,
#             [file_in[1]],
#             label_folder_in,
#             label_file_in,
#         )
#     N = len(nodes)

#     # Reading times
#     if times is None:
#         times = temporal_bt_times_read(
#             folder_in,
#             [file_in[2]],
#             label_folder_in,
#             label_file_in,
#         )
#     T = len(times)

#     # Read all possible paths
#     with open(file_out[0], 'rb') as fp:
#         all_paths = pickle.load(fp)

#     reached = []
#     r = len(all_paths)

#     # Check paths for hop-number or node-distance
#     c = 0
#     cmin = (r // 100) * (percentage - 1)
#     cmax = (r // 100) * (percentage)
#     for (p1, p2, t1, t2), val in all_paths.items():
#         if c < cmin:
#             continue
#         if c > cmax:
#             break
#         # Only run if cmin <= c <= cmax
#         d = val['d']
#         h = val['h']
#         # There is a path ...
#         if d < np.inf:
#             # And the path hop is more than 1
#             if h > 1:
#                 # Check the actual number of hop by filtering shortest path
#                 # OR if there is a path with shorter temporal distance
#                 sps = [
#                     np.array(p) % N for p in nx.all_shortest_paths(
#                         graph,
#                         source=(N * t1) + p1,
#                         target=(N * t2) + p2,
#                     )
#                 ]
#                 # Extract  intermediate nodes
#                 # And save in a dictionary based on path length
#                 spd = defaultdict(list)
#                 for x in sps:
#                     # Remove consecutive nodes (same node, next or previous time)
#                     temp = [
#                         v for i, v in enumerate(x) if i == 0 or v != x[i - 1]
#                     ]
#                     spd[len(temp)].append(temp)
#                 # Check only the shortest extracted paths
#                 h = min(spd.keys())
#                 spd = spd[h]
#                 # Minus 1 (-1) because hop length is 1 less than number of nodes
#                 all_paths[(p1, p2, t1, t2)]['h'] = h - 1
#                 # Unique intermediate nodes invole in shortest path
#                 paths = paths = set()
#                 for x in spd:
#                     paths.add(tuple(x[1:-1]))
#                 all_paths[(p1, p2, t1, t2)]['p'] = [list(x) for x in paths]

#     # Check if this is the last iteration
#     # Create a new reachability dataframe with updated hop count
#     if cmax == r:
#         reached = []
#         for (p1, p2, t1, t2), val in all_paths.items():
#             reached.append((p1, p2, t1, t2, val['d'], val['h']))
#         df = pd.DataFrame(reached, columns=['n1', 'n2', 't1', 't2', 'd'])
#         df.sort_values(
#             by=['n1', 'n2', 't1', 't2'],
#             inplace=True,
#             ignore_index=True,
#         )
#         if save:
#             df.to_csv(file_out[1], header=False, index=False)

#     if save:
#         with open(file_out[0], 'wb') as fp:
#             pickle.dump(all_paths, fp, protocol=pickle.HIGHEST_PROTOCOL)


def reachability_create(
    folder_in=NETWORK,
    folder_out=DB,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
        'bt_ton_weights.csv',
    ],
    file_out=[
        'uiuc.db',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    graph=None,
    times=None,
    nodes=None,
    ew=None,
    output=True,
    save=True,
):
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Reading graph
    if graph is None:
        graph = ton_bt_read(
            folder_in,
            [file_in[0]],
            label_folder_in,
            label_file_in,
        )

    # Reading nodes
    if nodes is None:
        nodes = temporal_bt_nodes_read(
            folder_in,
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )
    N = len(nodes)

    # Reading times
    if times is None:
        times = temporal_bt_times_read(
            folder_in,
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )
    T = len(times)

    # All existing network shortest path lenght (SPL)
    spl = dict(nx.all_pairs_shortest_path_length(graph))

    # Convert N1 -> N2 to Path of (Parent N1, Parent N2, t1, t2)
    # Save all reachable paths P(n1,n2,t1,t2) in a dataframe
    reached = []

    for n1 in spl:
        # Parent of source node
        p1 = n1 % N
        t1 = n1 // N
        for n2 in spl[n1]:
            p2 = n2 % N
            t2 = n2 // N
            # Nodes with different parents and timestamps
            if p1 != p2:
                # Add path lenght
                reached.append((p1, p2, t1, t2, spl[n1][n2]))

    # Create dataframe of reachability -> Easy to use for path filtering
    df = pd.DataFrame(reached, columns=['n1', 'n2', 't1', 't2', 'd'])
    # Then sort ...
    df.sort_values(
        by=['n1', 'n2', 't1', 't2'],
        inplace=True,
        ignore_index=True,
    )
    # Finally save reachability dataframe and Hop dictionary
    if save:
        df.to_sql(
            name='reachability',
            con=sqlite3.connect(file_out[0]),
            if_exists='replace',
            index_label='id'
        )

    # Check paths for hop-number or node-distance
    cc = 0  # Counter
    cr = 0  # Counter % passed
    cl = len(reached) // 100  # Counter limit = 1 % of all paths
    reached2 = []
    for p1, p2, t1, t2, d in reached:
        cc += 1
        if cc % cl == 0:
            cr += 1
            print(f'Processed {cr} % of all data ...')
        query = """ SELECT * FROM reachability WHERE n1 = ? AND n2 = ? AND t1 >= ? AND t2 <= ? ORDER BY d LIMIT 1; """
        try:
            con = sqlite3.connect(file_out[0])
            cur = con.cursor()
            cur.execute(query, (p1, p2, t1, t2))
            row = cur.fetchone()
            if row is not None:
                reached2.append((p1, p2, t1, t2, d, row[5]))
                # if d != row[5]:
                #     print((p1, p2, t1, t2, d, row[5]))
            cur.close()
        except sqlite3.Error as e:
            print(e)
        finally:
            con.close()

    # Create reachability again LOL
    df = pd.DataFrame(reached2, columns=['n1', 'n2', 't1', 't2', 'd', 'h'])
    reached2.clear()
    # Then sort ...
    df.sort_values(
        by=['n1', 'n2', 't1', 't2'],
        inplace=True,
        ignore_index=True,
    )
    if save:
        df.to_sql(
            name='reachability',
            con=sqlite3.connect(file_out[0]),
            if_exists='replace',
            index_label='id'
        )


def reachability_calculate(
    folder_in=NETWORK,
    folder_out=NETWORK,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
        'bt_ton_weights.csv',
    ],
    file_out=[
        'paths_all.p',
        'paths_all.csv',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    graph=None,
    times=None,
    nodes=None,
    ew=None,
    percentage=1,
    output=True,
    save=True,
):
    """
    Every time this function only work on 1% of the data from 1 .. 100
    """
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Reading graph
    if graph is None:
        graph = ton_bt_read(
            folder_in,
            [file_in[0]],
            label_folder_in,
            label_file_in,
        )

    # Reading nodes
    if nodes is None:
        nodes = temporal_bt_nodes_read(
            folder_in,
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )
    N = len(nodes)

    # Reading times
    if times is None:
        times = temporal_bt_times_read(
            folder_in,
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )
    T = len(times)

    # Read all possible paths
    with open(file_out[0], 'rb') as fp:
        all_paths = pickle.load(fp)

    reached = []
    r = len(all_paths)

    # Check paths for hop-number or node-distance
    c = 0
    cmin = (r // 100) * (percentage - 1)
    cmax = (r // 100) * (percentage)
    for (p1, p2, t1, t2), val in all_paths.items():
        if c < cmin:
            continue
        if c > cmax:
            break
        # Only run if cmin <= c <= cmax
        d = val['d']
        h = val['h']
        # There is a path ...
        if d < np.inf:
            # And the path hop is more than 1
            if h > 1:
                # Check the actual number of hop by filtering shortest path
                # OR if there is a path with shorter temporal distance
                sps = [
                    np.array(p) % N for p in nx.all_shortest_paths(
                        graph,
                        source=(N * t1) + p1,
                        target=(N * t2) + p2,
                    )
                ]
                # Extract  intermediate nodes
                # And save in a dictionary based on path length
                spd = defaultdict(list)
                for x in sps:
                    # Remove consecutive nodes (same node, next or previous time)
                    temp = [
                        v for i, v in enumerate(x) if i == 0 or v != x[i - 1]
                    ]
                    spd[len(temp)].append(temp)
                # Check only the shortest extracted paths
                h = min(spd.keys())
                spd = spd[h]
                # Minus 1 (-1) because hop length is 1 less than number of nodes
                all_paths[(p1, p2, t1, t2)]['h'] = h - 1
                # Unique intermediate nodes invole in shortest path
                paths = paths = set()
                for x in spd:
                    paths.add(tuple(x[1:-1]))
                all_paths[(p1, p2, t1, t2)]['p'] = [list(x) for x in paths]

    # Check if this is the last iteration
    # Create a new reachability dataframe with updated hop count
    if cmax == r:
        reached = []
        for (p1, p2, t1, t2), val in all_paths.items():
            reached.append((p1, p2, t1, t2, val['d'], val['h']))
        df = pd.DataFrame(reached, columns=['n1', 'n2', 't1', 't2', 'd'])
        df.sort_values(
            by=['n1', 'n2', 't1', 't2'],
            inplace=True,
            ignore_index=True,
        )
        if save:
            df.to_csv(file_out[1], header=False, index=False)

    if save:
        with open(file_out[0], 'wb') as fp:
            pickle.dump(all_paths, fp, protocol=pickle.HIGHEST_PROTOCOL)
