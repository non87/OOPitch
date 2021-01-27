'''
Deals with loading tracking and event data and translate them into object
At the moment, only works for Metrica data.

Code hevily reliant on https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking, by
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


import os
import zipfile
from io import BytesIO, StringIO
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict




def f24_2_SPDAL(f24, timestamps = None):
    '''
    This function translates the f24 xml to a pandas dataframe in the SPADL+ format. The logic of the translation is
    documented into a separate document. I added few categories to SPADL, that justifies the "+".

    :param f24: An xml.etree.ElementTree object, containing an Opta f24 file
    :return: A pandas.DataFrame containing the SPADL event for the match
    '''

    # Save the time periods start and end
    period_boundaries = []
    previous_event = None
    # it is necessary to keep track of end/ start period events because they are actually repeated in the f24!
    previous_event_end = False
    previous_event_start = False
    timestamps = []
    spadl = [[] for i in range(12)]
    print("Converting Events")
    for event in f24.findall('Event'):
        event_type = int(event.attrib['type_id'])
        if event_type in relevant:
            # print(f"relevant: {event_type}")
            # Start collecting data on the event
            timestamp = from_format(event.attrib['timestamp'], "YYYY-MM-DDTHH:mm:ss.SSS").timestamp()
            # There are some events for which the player seems unavailable. Not sure, but seems issues in the data
            # I do not register those
            try:
                player = event.attrib['player_id'] if event_type not in [30, 32, 37] else np.nan
            except:
                print(f"An event with no players. Id: {event.attrib['id']}")
                player = np.nan
                # We flag this
                event_type = -99
            x = float(event.attrib['x'])
            y = float(event.attrib['y'])
            team = event.attrib['team_id']
            # Get qualifiers id as integers and XML leaves
            quals = [(qual, int(qual.attrib['qualifier_id'])) for qual in event.findall("Q")]
            # Separate ids from leaves
            qual_leaf = {q[1]:q[0] for q in quals}
            quals = [q[1] for q in quals]
            # Update the spadl data! We have exception for the first iteration and period end/start events
            # We need to do this here because sometime we have to look ahead to complete an event
            if (previous_event is not None) and (previous_event not in [30, 32, 37, -99]):
                event_type, spadl = check_and_update(previous_event, event_type, spadl, new_row, timestamp, team, player, x, y, quals)
                # This makes sure we register only the first start/ end event of the sequence
                previous_event_end = False
                previous_event_start = False

            period = int(event.attrib['period_id'])
            previous_event = event_type

            new_row = [timestamp, period, player, team, x, y] # [spadl_event, x_end, y_end, outcome, body_part, special]

            # passes
            if event_type in [1, 2]:
                spadl_event, x_end, y_end, outcome, body_part, special = \
                    parse_passages(event, event_type=event_type, quals=quals, qual_leaf=qual_leaf)

            # Shooting
            elif event_type in [13, 14, 15, 16]:
                spadl_event, x_end, y_end, outcome, body_part, special = \
                    parse_shots(event, event_type=event_type, quals=quals, qual_leaf=qual_leaf)

            # Keeper
            elif event_type in [10, 11, 41, 52, 53, 54]:
                spadl_event, x_end, y_end, outcome, body_part, special = \
                        parse_keeper(event, event_type=event_type, quals=quals, qual_leaf=qual_leaf)
                previous_event_period = False

            # Foul
            elif event_type == 4:
                x_end, y_end, body_part, special = np.nan, np.nan, np.nan, np.nan
                outcome = "Failure"
                spadl_event = 'foul'

            # Tackle
            elif event_type == 7:
                new_row = [timestamp, period, player, team, x, y]
                x_end, y_end, body_part, special = np.nan, np.nan, np.nan, np.nan
                outcome = "Success" if event.attrib['outcome']==1 else "Failure"
                spadl_event = 'tackle'

            # Bad touch in SPADL is always connected to losing the ball.
            # Not here, but we still go with this
            elif (event_type == 61):
                x_end, y_end, body_part, special = np.nan, np.nan, np.nan, np.nan
                outcome = "Failure"
                spadl_event = 'bad touch'

            # Clearance
            elif (event_type == 12):
                x_end, y_end = float(qual_leaf[140].attrib['value']), float(qual_leaf[141].attrib['value'])
                # Header are signaled...I imagine if no body part qualifier the clearance was done with feet
                special, body_part = np.nan, 'either feet'
                if 15 in quals:
                    body_part = 'head'
                elif 72 in quals:
                    body_part = 'other'
                outcome = "Success"
                spadl_event = 'clearance'

            # Succesful Interception, I believe all interception in Opta have outcome 1
            elif (event_type == 8) and (event.attrib['outcome'] == "1"):
                x_end, y_end, body_part, special = np.nan, np.nan, np.nan, np.nan
                outcome = "Success"
                spadl_event = 'Intercept'

            # Take on
            elif (event_type == 3):
                x_end, y_end, body_part, special = np.nan, np.nan, np.nan, np.nan
                outcome = "Success" if event.attrib['outcome']==1 else "Failure"
                spadl_event = 'take on'

            # recovery/ this is not originally in SPADL
            elif (event_type == 49):
                x_end, y_end, body_part, special = np.nan, np.nan, np.nan, np.nan
                outcome = "Success"
                spadl_event = 'take on'


            # Periods-start-end. This fails if events are not ordered in the xml
            elif event_type in [30, 32, 37]:
                spadl_event = 'period'
                timestamp = from_format(event.attrib['timestamp'], "YYYY-MM-DDTHH:mm:ss.SSS").timestamp()
                if event_type == 32 and not previous_event_start:
                    period_boundaries.append(timestamp)
                elif event_type == 30 and not previous_event_end:
                    period_boundaries.append(timestamp)
                if event_type in [30, 37]:
                    previous_event_end = True
                else:
                    previous_event_start = True
                x_end = y_end = outcome = body_part = special = None

            # -99 are events that are helpful to check, but should not be registered.
            elif event_type == -99:
                # Make sure we do not register this
                spadl_event = 'flagged'
                x_end = y_end = outcome = body_part = special = None


            new_row.insert(1, spadl_event)
            new_row += [x_end, y_end, outcome, body_part, special]

    spadl = pd.DataFrame({'frame':spadl[0], 'event': spadl[1], 'period': spadl[2], 'player': spadl[3], 'team': spadl[4],
                                 'x_start': spadl[5], 'y_start':spadl[6], 'x_end': spadl[7], 'y_end': spadl[8],
                                 'outcome': spadl[9], 'body_part': spadl[10], 'special': spadl[11]})

    '''
    We need to align the frame in the tracking data with the timestamp in the f24 data
    this is not trivial because the timestamp is actually more precise than then frame (at hertz=25)
    we proceed by 
    1. aligning the kickoff frame (read from meta) with the kickoff timestamp (read from f24)
    2. calculate the difference in millisecond between successive events and the kickoff event
    3. round the difference to the closest frame
    '''
    print("Calculating Frames")
    assert len(period_boundaries) in [4,8]
    # Convert timestamps to frame
    start_time_frame = period_boundaries[0]
    # TODO mean of multiple estimators
    if timestamps is not None:
        timestamps = spadl['frame'] - start_time_frame
        frames = np.int_(np.round(timestamps * hertz)) + 1
    else:
        timestamps = spadl['frame'] - start_time_frame
        frames = np.int_(np.round(timestamps * hertz)) + 1
    spadl['frame'] = frames
    # We need to subtract the interval length from second period (and further) frames
    for i, boundary in enumerate(period_boundaries):
        if (i % 2 == 0) and (i > 0):
            interval_lenght_in_frame = np.int_(np.round((period_boundaries[i] - period_boundaries[i-1])*opta_hertz))
            period = (i / 2) + 1
            spadl.loc[spadl['period']== period, 'frame'] = spadl.loc[spadl['period']== period, 'frame'] - interval_lenght_in_frame

    return(spadl)


spadl = f24_2_SPDAL(f24)












zip_file_path = 'aalborg_recent.zip'
anonymized = False
os.chdir('/home/non1987/Documents/football_analysis')

zp = zipfile.ZipFile(zip_file_path)
f24 = [x for x in zp.filelist if 'f24' in x.filename][0]
f7  = [x for x in zp.filelist if 'f7' in x.filename][0]
trac = [x for x in zp.filelist if x.filename[-3:] == 'zip'][0]
# Read the zipped file inside the zip folder...
trac = zipfile.ZipFile(BytesIO(zp.read(trac)))
meta = [x for x in trac.filelist if 'meta' in x.filename][0]
trac_dt = [x for x in trac.filelist if 'meta' not in x.filename][0]
# Read the files
f24 = zp.read(f24)
f7 = zp.read(f7)
trac_dt = trac.read(trac_dt)
meta = trac.read(meta)
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

# Get player info from f7
f7 = ET.fromstring(f7)
# 0 = home, 1 = away
teams = {}
# Contain info on the teams. Helpful to have default if anonymized = True
teams_dt = {1: defaultdict(lambda: None, {}) , 0: defaultdict(lambda: None, {})}
# Contain info on the players for home and away
players_ids = {0:{}, 1:{}}
team_dt = f7.findall('SoccerDocument//TeamData')
# Not sure this is robust to neutral venues
for team in team_dt:
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
    for p in team.findall('PlayerLineUp//MatchPlayer'):
        j += 1
        if anonymized:
            #### CHECK THIS AGAIN
            if i == 0:
                base_str = 'away'
            else:
                base_str = 'home'
            p_id = '_'.join([base_str, str(j)])
            p_att = defaultdict(None, {'number': j, 'id': p_id})
        else:
            p_att = defaultdict(None, {'number': p.attrib['ShirtNumber'], 'id': p.attrib['PlayerRef']})
            player_id2numb[i][p.attrib['PlayerRef']] = int(p.attrib['ShirtNumber'])
        players_ids[i][int(p.attrib['ShirtNumber'])] = p_att

# We fill player and team data if needed
if anonymized == False:
    teams = f7.findall('SoccerDocument//Team')
    for team in teams:
        team_pos = teams_id[team.attrib['uID']]
        name = team.findall('Name')[0].text
        color = team.findall('Kit')[0].attrib['colour1']
        teams_dt[team_pos] = {'name': name, 'color': color}
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


trac_dt = pd.read_csv( StringIO(trac_dt.decode()), sep=':', header=0, names=['frame', 'player_dt', 'ball_dt', 'drop'])
trac_dt = trac_dt.drop(columns=['drop'])
# We use string methods below
trac_dt = trac_dt.astype({'frame': 'int64', 'player_dt': 'string', 'ball_dt': 'string'})

period_sel = np.zeros_like(trac_dt['frame'], dtype=np.bool)
for i, period in periods.items():
    period_sel[(trac_dt['frame'] >= period['start']) & (trac_dt['frame']<= period['end'])] = True
trac_dt = trac_dt.loc[period_sel]
# This creates a column for each humans on the pitch
player_dt = trac_dt['player_dt'].str.split(";", expand=True)
players = []
# Last column is empty
for col in player_dt.columns[:-1]:
    print(col)
    temp = player_dt[col].str.split(',', expand=True)
    temp.columns = ["team_id", "target_id", "jersey_no", "x", "y", "speed"]
    temp['frame'] = trac_dt['frame']
    players.append(temp)
del temp
# Get the data in a long format
player_dt = pd.concat(players, ignore_index=True)
del players

# To extract player id we identify the players by their team + jersey number
# It looks like home =1, away =0 here. Why? Is this common?
# Let's check with other matches later
def get_id(p_data, p_ids):
    return(p_ids[p_data['team_id']][p_data['jersey_no']]['id'])



player_dt = player_dt.loc[player_dt['team_id'].isin(["0","1"]),:]
player_dt = player_dt.astype({'team_id':'int8', 'jersey_no':'int16', 'x':'float64', 'y':'float64', 'speed': 'float64'})
player_dt['p_id'] = player_dt.apply(get_id, args=(players_ids,), axis=1)
player_dt = [player for p, player in player_dt.groupby('p_id')]
# Make sure the player_df are ordered
player_dt = [p_dt.sort_values('frame') for p_dt in player_dt]

player_trac_dt = trac_dt.iloc[:, 1].to_list()
# Add the frame
frame = trac_dt.iloc[:, 0].to_list()
player_trac_dt_l = np.array([len(x.split(";")) for x in player_trac_dt])


'''
Read f24 event file
'''

event_type_names = {
    1: "pass",
    2: "offside pass",
    3: "take on",
    4: "foul",
    5: "out",
    6: "corner awarded",
    7: "tackle",
    8: "interception",
    9: "turnover",
    10: "save",
    11: "claim",
    12: "clearance",
    13: "miss",
    14: "post",
    15: "attempt saved",
    16: "goal",
    17: "card",
    18: "player off",
    19: "player on",
    20: "player retired",
    21: "player returns",
    22: "player becomes goalkeeper",
    23: "goalkeeper becomes player",
    24: "condition change",
    25: "official change",
    26: "unknown26",
    27: "start delay",
    28: "end delay",
    29: "unknown29",
    30: "end",
    31: "unknown31",
    32: "start",
    33: "unknown33",
    34: "team set up",
    35: "player changed position",
    36: "player changed jersey number",
    37: "collection end",
    38: "temp_goal",
    39: "temp_attempt",
    40: "formation change",
    41: "punch",
    42: "good skill",
    43: "deleted event",
    44: "aerial",
    45: "challenge",
    46: "unknown46",
    47: "rescinded card",
    48: "unknown46",
    49: "ball recovery",
    50: "dispossessed",
    51: "error",
    52: "keeper pick-up",
    53: "cross not claimed",
    54: "smother",
    55: "offside provoked",
    56: "shield ball opp",
    57: "foul throw in",
    58: "penalty faced",
    59: "keeper sweeper",
    60: "chance missed",
    61: "ball touch",
    62: "unknown62",
    63: "temp_save",
    64: "resume",
    65: "contentious referee decision",
    66: "possession data",
    67: "50/50",
    68: "referee drop ball",
    69: "failed to block",
    70: "injury time announcement",
    71: "coach setup",
    72: "caught offside",
    73: "other ball contact",
    74: "blocked pass",
    75: "delayed start",
    76: "early end",
    77: "player off pitch",
}


from pendulum import from_format

f24 = ET.fromstring(f24)
f24 = f24.find('Game')



# These columns seem to work
# le dimensioni x,y sono in cm.
# le jersey no vanno collegate con f7
# target id non so cosa sia
# team id: 4 non so cosa sia; 0 = home ("H"), 1 = away ("A")

from datetime import datetime

datetime.strptime("2020-12-20T15:02:40.75", "%Y-%m-%dT%H:%M%:%S.%f")
datetime.strptime('2020-12-20T15:02:40.75', "%Y-%m-%dT%H:%M:%S.%f")
pendulum.from_format("2020-12-20T15:02:40.422", "YYYY-MM-DDTHH:mm:ss.SSS")
dt = from_format("2020-12-20T15:02:40.422", "YYYY-MM-DDTHH:mm:ss.SSS")
dt.timestamp()
'2020-12-20T15:03:09.422'

datetime.datetime

EVENT_QUALIFIER_GOAL_KICK = 124
EVENT_QUALIFIER_FREE_KICK = 5
EVENT_QUALIFIER_THROW_IN = 107
EVENT_QUALIFIER_CORNER_KICK = 6
EVENT_QUALIFIER_PENALTY = 9
EVENT_QUALIFIER_KICK_OFF = 279

#qualifier 140, 141 x,y of pass receiver
# qualifier 102, 103 x y of goal mouth. 102 seems what we are looking for. 54 = palo sinistro, 47 = palo destra, circa
# 0 sara' il minimo di 103. Il massimo (dentro la porta) sembra 41
# qualifier 146, 147 x, y (goal mouth) of blocked shot
# qualifier 230, 231 x, y of blocked shots?
# qualifier 31, 32 and 33: yellow, second yellow, red


min_dribble_length: float = 3.0
max_dribble_length: float = 60.0
max_dribble_duration: float = 10.0


def _add_dribbles(actions: DataFrame) -> DataFrame:
    next_actions = actions.shift(-1)

    same_team = actions.team_id == next_actions.team_id
    # not_clearance = actions.type_id != actiontypes.index("clearance")

    dx = actions.end_x - next_actions.start_x
    dy = actions.end_y - next_actions.start_y
    far_enough = dx ** 2 + dy ** 2 >= min_dribble_length ** 2
    not_too_far = dx ** 2 + dy ** 2 <= max_dribble_length ** 2

    dt = next_actions.time_seconds - actions.time_seconds
    same_phase = dt < max_dribble_duration
    same_period = actions.period_id == next_actions.period_id

    dribble_idx = same_team & far_enough & not_too_far & same_phase & same_period

    dribbles = pd.DataFrame()
    prev = actions[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles['game_id'] = nex.game_id
    dribbles['period_id'] = nex.period_id
    dribbles['action_id'] = prev.action_id + 0.1
    dribbles['time_seconds'] = (prev.time_seconds + nex.time_seconds) / 2
    if 'timestamp' in actions.columns:
        dribbles['timestamp'] = nex.timestamp
    dribbles['team_id'] = nex.team_id
    dribbles['player_id'] = nex.player_id
    dribbles['start_x'] = prev.end_x
    dribbles['start_y'] = prev.end_y
    dribbles['end_x'] = nex.start_x
    dribbles['end_y'] = nex.start_y
    dribbles['bodypart_id'] = spadlconfig.bodyparts.index('foot')
    dribbles['type_id'] = spadlconfig.actiontypes.index('dribble')
    dribbles['result_id'] = spadlconfig.results.index('success')

    actions = pd.concat([actions, dribbles], ignore_index=True, sort=False)
    actions = actions.sort_values(['game_id', 'period_id', 'action_id']).reset_index(drop=True)
    actions['action_id'] = range(len(actions))
    return actions
©