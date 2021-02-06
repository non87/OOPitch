'''
Deals with loading tracking and event data and translate them into object
At the moment, only works for Metrica data.

Code heavily reliant on https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking, by
Laurie Shaw  (@EightyFivePoint)
'''

import pandas as pd
import csv as csv
import numpy as np
from shapely_football import Point
from OOPitch import Player, Ball, Pitch, Match
import matplotlib.pyplot as plt
import urllib
from shapely.affinity import affine_transform, rotate
from opta_utils import *
import os
import zipfile
from io import BytesIO, StringIO
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict


meters_per_yard = 0.9144  # unit conversion from yards to meters
metrica_hertz = 25  # The frequency per second of the tracking data in Metrica
opta_hertz = 25


def create_Point(x, y):
    '''
    Create a OOPitch Point from x and y while controlling for NA.

    :param x: x coordinate
    :param y: y coordinate
    :return: a OOPitch Point from the x,y coordinates if neither x nor y are Nan. Nan otherwise
    '''
    if not np.isnan(x) and not np.isnan(x):
        return Point([x, y])
    return np.nan


def change_metrica_coor(p, pitch_dim=(106, 68)):
    '''
    Convert positions from Metrica units to meters (with origin at centre circle).


    :param p: The Point object to be converted
    :param pitch_dim: An iterable containing the dimension of the pitch in meters, first length, than width
    :return: A Point object with meter coordinate
    '''
    if isinstance(p, Point):
        length, width = [dim for dim in pitch_dim]
        return Point(affine_transform(p, [length, 0, 0, width, -length / 2, -width / 2]))
        # At the moment it is unclear if I this is sufficient or I need to reflect the y
        # return affine_transform(p, [0, lenght, 0, -width, -lenght / 2, width / 2])
    else:
        return np.nan

def change_direction(x):
    '''
    Flip coordinates in second half so that each team always shoots in the same direction through the match.
    '''
    if isinstance(x, Point):
        return Point(rotate(x, 180, origin=Point((0, 0))))
    else:
            return np.nan



def read_metrica_tracking(data_path, teamname, get_ball=True, pitch_dim=(106, 68)):
    '''
    read Metrica tracking data for game_id and return as a list of Player (and Ball) objects.

    :param data_path: str (or valid path). Where is the tracking data. Can also be url to the data (no parameters can
        be passed in the request).
    :param teamname: str. What team is the data about?
    :param get_ball: Boolean. Whether the ball data should be used to create a Ball object or ignored.
    :return: A list containing Player objects and a Ball object. No guarantee on the order of the objects in the list.
    '''
    try:
        csvfile = open(data_path, 'r')
        reader = csv.reader(csvfile)
        local = True
    except FileNotFoundError:
        try:
            dt = urllib.request.urlopen(data_path)
            # Decode first 3 rows
            lines = [dt.readline().decode('utf-8') for i in range(3)]
            reader = csv.reader(lines)
            local = False
        except urllib.error.URLError:
            raise FileNotFoundError(f'No file at {data_path}')
    # First:  deal with file headers so that we can get the player names correct
    print(f"Reading team: {teamname}")
    # We don't do much with this, but still need to get the first row out
    teamnamefull = next(reader)[3].lower()
    # construct column names
    jerseys = [x for x in next(reader) if x != '']  # extract player jersey numbers from second row
    columns = next(reader)
    if local: csvfile.close()
    # Keep track of the x-y couples of columns
    columns_couples = {'ball': ['ball_x', 'ball_y']}  # column headers for the x & y positions of the ball
    for i, j in enumerate(jerseys):  # create x & y position column headers for each player
        base = "{}_{}".format(teamname, j)
        x_name = "_".join([base, "x"])
        y_name = "_".join([base, "y"])
        columns[i * 2 + 3] = x_name
        columns[i * 2 + 4] = y_name
        columns_couples[base] = [x_name, y_name]
    columns[-2] = columns_couples['ball'][0]
    columns[-1] = columns_couples['ball'][1]
    # Second: read in tracking data and place into pandas Dataframe
    tracking = pd.read_csv(data_path, names=columns, index_col='Frame', skiprows=3)
    # Check when the Periods starts. Takes into account possible overtime
    period_change = tracking.index[tracking['Period'].diff(periods=1) > 0].values
    # Third: create shapely.Point
    players = []
    for couple_name, couple_columns in columns_couples.items():
        if couple_name == "ball" and get_ball:
            print("Creating Point objects for %s" % couple_name)
            # Create points, convert to meters
            point_data = tracking.apply(lambda row: change_metrica_coor(create_Point(row[couple_columns[0]],
                                                                                     row[couple_columns[1]]),
                                                                        pitch_dim=pitch_dim), axis=1)
            # Invert the direction in second half
            # Check out how over-time is treated in the future. This may give unexpected behavior in that case
            point_data.loc[tracking['Period'] > 1] = point_data.loc[tracking['Period'] > 1].apply(change_direction)

            tracking[couple_name] = point_data
            # Fill a 2 seconds hole if needed // makes it robust to random missing data in the middle
            tracking[couple_name] = tracking[couple_name].fillna(method='bfill', limit=40)
            players.append(Ball(positions=tracking[couple_name], hertz=metrica_hertz))
        elif couple_name == "ball":
            pass
        else:
            print("Creating Point objects for %s" % couple_name)
            # Create points, convert to meters
            point_data = tracking.apply(lambda row: change_metrica_coor(create_Point(row[couple_columns[0]],
                                                                                     row[couple_columns[1]]),
                                                                        pitch_dim=pitch_dim), axis=1)
            point_data.loc[tracking['Period'] > 1] = point_data.loc[tracking['Period'] > 1].apply(change_direction)
            tracking[couple_name] = point_data
            # Fill a 2 seconds hole if needed // makes it robust to random missing data in the middle
            tracking[couple_name] = tracking[couple_name].fillna(method='bfill', limit=40)
            players.append(Player(couple_name, teamname, number=couple_name[-2:], positions=tracking[couple_name],
                                  hertz=metrica_hertz))
    # Need to do the ball as well
    return players, period_change


def read_metrica_events(data_path, pitch_dim):
    '''
    read_event_data(DATADIR,game_id):
    read Metrica event data  for game_id and return as a DataFrame
    For now we stick to this basic event data frame, but eventually, we will pass to SPDAL and atomic-SPADL format
    '''
    events = pd.read_csv(data_path)  # read data
    # ge the Point columns
    events['Start'] = events.apply(lambda row: change_metrica_coor(create_Point(row["Start X"], row["Start Y"]),
                                                                   pitch_dim=pitch_dim), axis=1)

    events['End'] = events.apply(lambda row: change_metrica_coor(create_Point(row["End X"], row["End Y"]),
                                                                 pitch_dim=pitch_dim), axis=1)
    # applymap seems to work. But it may be unstable in future version of pandas
    # Check out how over-time is treated in the future. This may give unexpected behavior in that case
    events.loc[events['Period'] > 1, ['Start', 'End']] = events.loc[events['Period']>1, ['Start', 'End']].applymap(change_direction)

    return events


def read_metrica(home_path, away_path, events_path, home_name='home', away_name='away', pitch_dimen=(106.0, 68.0),
                 n_grid_cells_x=50):
    '''
    Function that creates a match object starting from complete Metrica data for the match. Complete data encompasses
    three files: an event dataset, a home-tracking dataset and an away tracking dataset. The function needs the path
    to those three files.

    :param home_path: Path to the home tracking data
    :param away_path: Path to the away tracking data
    :param events_path: Path to the event data
    :param home_name: Name of the team whose tracking data is at home_path
    :param away_name: Name of the team whose tracking data is at home_path
    :param field_dimen: Dimension of the pitch
    :param n_grid_cells_x: Number of subptich for the Pitch object
    :return: A match object from the data
    '''
    home_tracking, halves = read_metrica_tracking(home_path, home_name, get_ball=True)
    # Get the ball take it out of the home tracking data
    ball_n = [i for i, obj in enumerate(home_tracking) if isinstance(obj, Ball)]
    ball = home_tracking.pop(ball_n[0])
    away_tracking, _ = read_metrica_tracking(away_path, away_name, get_ball=False)
    # Read event data
    events = read_metrica_events(events_path, pitch_dimen)
    print('Read events data')
    # Create the pitch
    pitch = Pitch(pitch_dimen=pitch_dimen, n_grid_cells_x=n_grid_cells_x)
    return Match(home_tracking, away_tracking, ball, events, pitch, halves=halves, hertz=metrica_hertz)



def open_chyron_zip(zip_file_path):
    zp = zipfile.ZipFile(zip_file_path)
    f24 = [x for x in zp.filelist if 'f24' in x.filename][0]
    f7  = [x for x in zp.filelist if 'f7' in x.filename][0]
    trac = [x for x in zp.filelist if x.filename[-3:] == 'zip'][0]
    # Read the zipped file inside the zip folder...
    trac = zipfile.ZipFile(BytesIO(zp.read(trac)))
    meta = [x for x in trac.filelist if 'meta' in x.filename][0]
    trac_dt = [x for x in trac.filelist if 'meta' not in x.filename][0]
    f24 = zp.read(f24)
    f7 = zp.read(f7)
    trac_dt = trac.read(trac_dt)
    meta = trac.read(meta)
    return(f7, f24, meta, trac_dt)

def parse_meta_chyron(meta):
    # Read the files
    meta = ET.fromstring(meta)
    # Get match_info from meta
    meta = meta.find('match')
    hertz = int(meta.attrib["iFrameRateFps"])
    pitch_dim = ( int(float(meta.attrib["fPitchXSizeMeters"])), int(float(meta.attrib["fPitchYSizeMeters"])))
    # Important to cut and normalize frames
    periods = {}
    for period in meta.findall('period'):
        if int(period.attrib['iStartFrame']) > 0:
            period_dict = {'start': int(period.attrib['iStartFrame']), 'end': int(period.attrib['iEndFrame'])}
            periods[period.attrib['iId']] = period_dict
    return(periods, pitch_dim, hertz)

def parse_f7_opta(f7, anonymized = False):
    # Get player info from f7
    f7 = ET.fromstring(f7)
    # 0 = home, 1 = away
    teams = {}
    # Contain info on the teams. Helpful to have default if anonymized = True
    teams_dt = {1: defaultdict(lambda: None, {}) , 0: defaultdict(lambda: None, {})}
    # Contain info on the players for home and away
    players_ids = {0:{}, 1:{}}
    team_xml = f7.findall('SoccerDocument/MatchData/TeamData')
    # Not sure this is robust to neutral venues
    for team in team_xml:
        if team.attrib['Side'] == 'Home':
            ###### CHECK THIS AGAIN
            teams[1] = team
        else:
            teams[0] = team
    # Useful to get the players' names
    teams_id = {teams[0].attrib['TeamRef']: 0, teams[1].attrib['TeamRef']: 1}
    player_id2numb = {0:{}, 1:{}}

    for i, team in teams.items():
        j = 0
        if anonymized:
            if team.attrib['Side'] == "Home":
                base_str = 'home'
                teams_dt[i]['anon_id'] = 'home'
                teams_dt[i]['venue'] = 'home'
                teams_dt[i]['id'] = team.attrib['TeamRef']
                teams_dt[i]['name'] = 'home'
            else:
                base_str = 'away'
                teams_dt[i]['anon_id'] = 'away'
                teams_dt[i]['venue'] = 'away'
                teams_dt[i]['id'] = team.attrib['TeamRef']
                teams_dt[i]['name'] = 'away'
        for p in team.findall('PlayerLineUp/MatchPlayer'):
            j += 1
            if anonymized:
                p_id = '_'.join([base_str, str(j)])
                p_att = defaultdict(lambda: None, {'number': j, 'anon_id': p_id, 'id': p.attrib['PlayerRef']})
            else:
                p_att = defaultdict(lambda: None, {'number': p.attrib['ShirtNumber'], 'id': p.attrib['PlayerRef']})
                player_id2numb[i][p.attrib['PlayerRef']] = int(p.attrib['ShirtNumber'])
            players_ids[i][int(p.attrib['ShirtNumber'])] = p_att

    # We fill player and team data if needed
    teams = f7.findall('SoccerDocument//Team')
    for i, team in enumerate(teams):
        if anonymized == False:
            team_pos = teams_id[team.attrib['uID']]
            team_id = team.attrib['uID']
            name = team.findall('Name')[0].text
            color = team.findall('Kit')[0].attrib['colour1']
            venue = 'home' if i == 1 else 'away'
            teams_dt[team_pos] = defaultdict(lambda: None, {'id': team_id, 'name': name, 'color': color, 'venue': venue})
            players = team.findall('Player')
            for player in players:
                p_id = player.attrib['uID']
                p_numb = player_id2numb[team_pos][p_id]
                # We use known as name; if None, lastname
                p_name = player.find('PersonName//Known')
                if p_name is None:
                    p_name = player.find('PersonName//Last').text
                else:
                    p_name = p_name.text
                # Put the name in place
                players_ids[team_pos][p_numb]['name'] = p_name
    return(teams_dt, players_ids)

def parse_f24_opta(f24, pitch_dim=(106, 68), teams=None, periods=None, players=None, anonymized=False, hertz=25):
    if anonymized and ((players is None) or (teams is None)):
        raise ValueError("A players and a teams dictionaries are needed to anonymize data")
    f24 = ET.fromstring(f24)
    f24 = f24.find("Game")
    spadl = f24_2_SPDAL(f24, periods=periods, hertz=hertz)
    spadl = spadl.astype({'team':str, 'player':str, 'frame':int})
    spadl['player'] = 'p' + spadl['player']
    spadl['team'] = 't' + spadl['team']
    if teams is None:
        home_team = spadl["team"].unique()[0]
        away_team = spadl["team"].unique()[1]
    else:
        for i,team in teams.items():
           if team['venue'] == 'home':
               home_team = team['id']
           elif team['venue'] == 'away':
               away_team = team['id']
    # Anonymize the data
    if anonymized:
        for team in teams.values():
            spadl.loc[spadl['team'] == team['id'], 'team'] = team['anon_id']
        for team in players.keys():
            for player in players[team].values():
                spadl.loc[spadl['player'] == player['id'], 'player'] = player['anon_id']


    # Make sure home attack from right to left and convert to meter using simple linear transformations
    to_meter = np.diag(pitch_dim)/100
    # Re-center on the center of the pitch
    spadl[['x_start', 'y_start','x_end', 'y_end']] = spadl[['x_start', 'y_start','x_end', 'y_end']] - 50
    # Minus sign: reflection across the origin
    starts = [- spadl.loc[spadl['team'] == home_team, ['x_start', 'y_start']] @ to_meter]
    starts.append(spadl.loc[spadl['team'] == away_team, ['x_start', 'y_start']] @ to_meter)
    spadl[['x_start', 'y_start']] = pd.concat(starts)
    # Same things with x_end, y_end (separated to not propagate nans)
    ends = [- spadl.loc[spadl['team'] == home_team, ['x_end', 'y_end']] @ to_meter]
    ends.append(spadl.loc[spadl['team'] == away_team, ['x_end', 'y_end']] @ to_meter)
    spadl[['x_end', 'y_end']] = pd.concat(ends)
    return(spadl)

def parse_tracab(trac_dt, periods, teams, players_id, spadl=None, anonymized=False):
    '''
    Parse the .dat tracab file in a more convenient pandas dataframe.

    :param trac_dt: The trac_dt file path
    :param periods: A periods dictionary from parse_meta_chyron
    :param teams: A dictionary containing team data. Result of parse_f7()
    :param players_id: A dictionary containing player data. Result of parse_f7()
    :param spadl: A spadl event dataframe as obtained from a parse_f24(). Optional
    :param anonymized: Return anonymized data if True. Default False.
    :return: A dictionary of players tracking data DataFrame and a ball tracking DataFrame
    '''


    # To extract player id we identify the players by their team + jersey number
    # It looks like home =1, away =0 here. Why? Is this common?
    # Let's check with other matches later
    # TODO Check that team_id == 1 is away in every match
    def get_id(p_data, p_ids, anonymized=False):
        if not anonymized:
            p_data['team_id']
            p_data['jersey_no']
            try:
                p_ids[p_data['team_id']]
            except KeyError:
                print(p_ids)
                print(p_data['team_id'])
            return (p_ids[p_data['team_id']][p_data['jersey_no']]['id'])
        else:
            return (p_ids[p_data['team_id']][p_data['jersey_no']]['anon_id'])

    print("Parsing tracking data")
    trac_dt = pd.read_csv(trac_dt, sep=':', header=0, names=['frame', 'player_dt', 'ball_dt', 'drop'])
    trac_dt = trac_dt.drop(columns=['drop'])
    # We use string methods below
    trac_dt = trac_dt.astype({'frame': 'int64', 'player_dt': 'string', 'ball_dt': 'string'})
    # Select frames that are within halves
    period_sel = np.zeros_like(trac_dt['frame'], dtype=np.bool)
    for i, period in periods.items():
        # We check we do not erase frames that are mentioned in the spadl event data
        if spadl is not None:
            start = np.min((period['start'], spadl.loc[spadl['period'] == int(i), "frame"].min()))
            # We add an extra second given that the last event in the spadl is not the end of the period.
            end = np.max((period['start'], spadl.loc[spadl['period'] == int(i), "frame"].max())) + 25
        else:
            start = period['start']
            end = period['end']
        period_sel[(trac_dt['frame'] >= start) & (trac_dt['frame'] <= end)] = True
    trac_dt = trac_dt.loc[period_sel]
    # This creates a column for each humans on the pitch
    player_dt = trac_dt['player_dt'].str.split(";", expand=True)
    players = []
    # Last column is empty
    for col in player_dt.columns[:-1]:
        print(f"Elaborating player number {col+1}")
        temp = player_dt[col].str.split(',', expand=True)
        temp.columns = ["team_id", "target_id", "jersey_no", "x", "y", "speed"]
        temp['frame'] = trac_dt['frame']
        players.append(temp)
    del temp, col
    # Get the data in a long format
    player_dt = pd.concat(players, ignore_index=True)
    del players
    # Only players in the two teams (no referee)
    player_dt = player_dt.loc[player_dt['team_id'].isin(["0","1"]),:]
    player_dt = player_dt.astype({'team_id':'int8', 'jersey_no':'int8', 'x':'float64', 'y':'float64', 'speed': 'float64'})
    player_dt['p_id'] = player_dt.apply(get_id, args=(players_id,), anonymized=anonymized, axis=1)
    # Convert to meter. For the rest, the coordinates should already be in the right format
    player_dt['x'] = player_dt['x']/100
    player_dt['y'] = player_dt['y']/100
    # Change team name
    if anonymized:
        player_dt.loc[player_dt['team_id'] == 0, 'team_id'] = teams[0]['anon_id']
        player_dt.loc[player_dt['team_id'] == 1, 'team_id'] = teams[1]['anon_id']
    else:
        player_dt.loc[player_dt['team_id'] == 0, 'team_id'] = teams[0]['id']
        player_dt.loc[player_dt['team_id'] == 1, 'team_id'] = teams[1]['id']
    # Reoder
    player_dt = player_dt[['frame', 'p_id', 'team_id', 'x', 'y', 'speed', 'jersey_no']]
    player_dt = {p:player for p, player in player_dt.groupby('p_id')}
    # Make sure the player_df are ordered
    player_dt = {p:p_dt.sort_values('frame') for p, p_dt in player_dt.items()}

    # Now the ball
    print("Elaborating ball")
    trac_dt['ball_dt'] = trac_dt['ball_dt'].str.strip(';')
    ball = trac_dt['ball_dt'].str.split(",", expand=True)
    ball = ball.drop(columns=[6])
    ball = ball.rename(columns={0: "x", 1: "y", 2: "z", 3: "speed", 4: "owning_team", 5: "state"})
    ball['frame'] = trac_dt['frame']
    # Reorder
    ball = ball[['frame', 'x', 'y', 'z', 'speed', 'owning_team', 'state']]
    ball = ball.astype({'frame': 'int32', 'x': 'float64', 'y': 'float64', 'z': 'float64', 'speed': 'float64',
                        'owning_team': 'category', 'state': 'category'})
    # To meter
    ball['x'] = ball['x']/100
    ball['y'] = ball['y']/100
    ball['z'] = ball['z']/100
    ball['speed'] = ball['speed']/100
    # Rename teams
    if anonymized:
        ball['owning_team'].cat.rename_categories({'A': "away", 'H': "home"})
    else:
        id_home = teams[1]['id'] if teams[1]['venue'] == 'home' else teams[0]['id']
        id_away = teams[1]['id'] if teams[1]['venue'] == 'away' else teams[0]['id']
        ball['owning_team'] = ball['owning_team'].cat.rename_categories({'A': id_away, 'H': id_home})
    return(player_dt, ball)

def read_chyronego(zip_file_path=None, f24_path=None, f7_path=None, tracking_path=None, meta_path=None,
                   anonymized=False, n_grid_cells_x=50):
    '''
    Read the event and tracking files related to a match from the Chyronego portal and return a match object.
    The files can be contained in one zipped folder -- as downloaded from the Chyronego web-portal -- or being stored
    separately. In the latter case, the function needs a path for each of the f24, f7, meta and tracking file.

    :param zip_file_path: The folder where the match data (f24, f7, meta and tracking) is stored
    :param f24_path, f7_path, tracking_path, meta_path: path to the different files containing the match data.
       Ignored if zip_file_path is specified
    :param anonymized: Should the match file contain anonymized data?
    :param n_grid_cells_x: Number of cells dividing the pitch object.
    :return: A match object containing the data for the match
    '''
    # For zipped folders from the chyronego portal
    if zip_file_path is not None:
        f7, f24, meta, trac_dt = open_chyron_zip(zip_file_path)
        trac_dt = StringIO(trac_dt.decode())
    # In case files are stored separately
    elif (f24_path is not None) and (f7_path is not None) and (tracking_path is not None) and (meta_path is not None):
        with open(f24_path, 'r') as fl:
            f24 = fl.read()
        with open(f7_path, 'r') as fl:
            f7 = fl.read()
        with open(meta_path, 'r') as fl:
            meta = fl.read()
        trac_dt = tracking_path
    else:
        raise ValueError("If zip_file_path is not given, all of f24_path, f7_path and tracking_path must be passed")

    periods, pitch_dim, hertz = parse_meta_chyron(meta)
    teams, players = parse_f7_opta(f7, anonymized=anonymized)
    home_team = 0 if teams[0]['venue'] == 'home' else 1
    away_team = int(np.abs(home_team - 1))
    spadl = parse_f24_opta(f24, teams=teams, pitch_dim=pitch_dim, periods=periods)
    players_dt, ball_dt = parse_tracab(trac_dt, periods, teams, players, anonymized=anonymized, spadl=spadl)
    # Create Points instead from x, y columns coordinate.
    id_k = 'id' if not anonymized else 'anon_id'
    ids_by_teams = [[],[]]
    for t in [away_team, home_team]: ids_by_teams[t] =[p[id_k] for p in players[t].values()]
    players_object = [[], []]
    re_frame = pd.Series(np.arange(1, ball_dt.shape[0] + 1))
    re_frame.index = ball_dt['frame']
    # Get the first frame of each halves (a part the first)
    halves = np.where(ball_dt['frame'].diff(1) > 1)[0] + 1
    halves = halves.tolist()
    for p_id, p in players_dt.items():
        print(f"Creating Point objects for {p_id}")
        team = home_team if p_id in ids_by_teams[home_team] else away_team
        jersey = p['jersey_no'].unique()[0]
        # Change Frames
        p = p.set_index('frame')
        p['frame'] = re_frame
        p = p.set_index('frame')
        # Change attacking direction in 2nd halves in extra times.
        # TODO check rules for overtime
        for i, half in enumerate(halves):
            if i % 2 == 0:
                if i < len(halves)-1:
                    next_half = halves[i+1]-1
                else:
                    next_half = np.inf
                p.loc[half:next_half, 'x'] = - p.loc[half:next_half, 'x']
                p.loc[half:next_half, 'y'] = - p.loc[half:next_half, 'y']
        positions = p.apply(lambda x: create_Point(x['x'], x['y']), axis=1)
        positions = positions.reindex(re_frame)
        players_object[team].append(Player(player_id=p_id, team=teams[team]['name'], positions=positions,
                                           number=players[team][jersey]['number'], hertz=hertz))

    ball_dt = ball_dt.set_index(ball_dt['frame'])
    ball_dt['frame'] = re_frame
    ball_dt = ball_dt.set_index(ball_dt['frame'])
    # Change attack side for the ball
    for i, half in enumerate(halves):
        if i % 2 == 0:
            if i < len(halves) - 1:
                next_half = halves[i + 1] - 1
            else:
                next_half = np.inf
            ball_dt.loc[half:next_half, 'x'] = - ball_dt.loc[half:next_half, 'x']
            ball_dt.loc[half:next_half, 'y'] = - ball_dt.loc[half:next_half, 'y']

    ball_point = ball_dt.apply(lambda x: create_Point(x['x'], x['y']), axis=1)
    ball = Ball(positions=ball_point, hertz=hertz, in_play=ball_dt['state'])
    # Check colors
    if teams[home_team]['color'] is not None:
        colors = (teams[home_team]['color'], teams[away_team]['color'])
    else:
        colors = ('red', 'blue')

    spadl = spadl.set_index('frame')
    spadl['frame'] = re_frame
    spadl = spadl.set_index(np.arange(1, spadl.shape[0] + 1))
    # Change attack side for the events
    for i, half in enumerate(halves):
        if i % 2 == 0:
            if i < len(halves) - 1:
                next_half = halves[i + 1] - 1
            else:
                next_half = np.inf
            relevant = (spadl['frame']>=half) & (spadl['frame']<=next_half)
            spadl.loc[relevant, 'x_start'] = - spadl.loc[relevant, 'x_start']
            spadl.loc[relevant, 'y_start'] = - spadl.loc[relevant, 'y_start']
            spadl.loc[relevant, 'x_end'] = - spadl.loc[relevant, 'x_end']
            spadl.loc[relevant, 'y_end'] = - spadl.loc[relevant, 'y_end']
    spadl['start'] = spadl.apply(lambda x: create_Point(x['x_start'], x['y_start']), axis=1)
    spadl['end'] = spadl.apply(lambda x: create_Point(x['x_end'], x['y_end']), axis=1)
    # Re order and drop
    spadl = spadl[['frame', 'event', 'period', 'player', 'team', 'outcome', 'start', 'end', 'body_part', 'special']]
    # Create the pitch
    pitch = Pitch(pitch_dimen=pitch_dim, n_grid_cells_x=n_grid_cells_x)
    match = Match(home_tracking=players_object[home_team], away_tracking=players_object[away_team], events=spadl, ball=ball,
                  pitch=pitch, halves=halves, away_name=teams[away_team]['name'],
                  home_name=teams[home_team]['name'], hertz=hertz, colors=colors)
    return(match)



