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

meters_per_yard = 0.9144  # unit conversion from yards to meters
metrica_hertz = 25  # The frequency per second of the tracking data in Metrica


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