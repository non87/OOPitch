'''
Deals with loading tracking and event data and translate them into object
At the moment, only works for Metrica data.

Code hevily reliant on https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking, by
Laurie Shaw  (@EightyFivePoint)
'''

import pandas as pd
import csv as csv
import numpy as np
from shapely_footbal import Point
from OOPitch import Player, Ball, Pitch, Match
import matplotlib.pyplot as plt
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

    :param data_path: str (or valid path). Where is the tracking data
    :param teamname: str. What team is the data about?
    :param get_ball: Boolean. Whether the ball data should be used to create a Ball object or ignored.
    :return: A list containing Player objects and a Ball object. No guarantee on the order of the objects in the list.
    '''
    # First:  deal with file headers so that we can get the player names correct
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        teamnamefull = next(reader)[3].lower()
        print("Reading team: %s" % teamname)
        # construct column names
        jerseys = [x for x in next(reader) if x != '']  # extract player jersey numbers from second row
        columns = next(reader)
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
            players.append(Ball(couple_name, positions=tracking[couple_name], hertz=metrica_hertz))
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
    return tracking, period_change


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


# def save_match_clip(hometeam, awayteam, fpath, fname='clip_test', figax=None, frames_per_second=25,
#                     team_colors=('r', 'b'), field_dimen=(106.0, 68.0), include_player_velocities=False,
#                     PlayerMarkerSize=10, PlayerAlpha=0.7):
#     """ save_match_clip( hometeam, awayteam, fpath )
#
#     Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'
#
#     Parameters
#     -----------
#         hometeam: home team tracking data DataFrame. Movie will be created from all rows in the DataFrame
#         awayteam: away team tracking data DataFrame. The indices *must* match those of the hometeam DataFrame
#         fpath: directory to save the movie
#         fname: movie filename. Default is 'clip_test.mp4'
#         fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
#         frames_per_second: frames per second to assume when generating the movie. Default is 25.
#         team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
#         field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
#         include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
#         PlayerMarkerSize: size of the individual player marlers. Default is 10
#         PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
#
#     Returrns
#     -----------
#        fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
#
#     """
#     # check that inanimation.writers['ffmpeg']dices match first
#     assert np.all(hometeam.index == awayteam.index), "Home and away team Dataframe indices must be the same"
#     # in which case use home team index
#     index = hometeam.index
#     # Set figure and movie settings
#     FFMpegWriter = animation.writers['ffmpeg']
#     metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
#     writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
#     fname = fpath + '/' + fname + '.mp4'  # path and filename
#     # create football pitch
#     if figax is None:
#         fig, ax = plot_pitch(field_dimen=field_dimen)
#     else:
#         fig, ax = figax
#     fig.set_tight_layout(True)
#     # Generate movie
#     print("Generating movie...", end='')
#     with writer.saving(fig, fname, 100):
#         for i in index:
#             figobjs = []  # this is used to collect up all the axis objects so that they can be deleted after each iteration
#             for team, color in zip([hometeam.loc[i], awayteam.loc[i]], team_colors):
#                 x_columns = [c for c in team.keys() if
#                              c[-2:].lower() == '_x' and c != 'ball_x']  # column header for player x positions
#                 y_columns = [c for c in team.keys() if
#                              c[-2:].lower() == '_y' and c != 'ball_y']  # column header for player y positions
#                 objs, = ax.plot(team[x_columns], team[y_columns], color + 'o', MarkerSize=PlayerMarkerSize,
#                                 alpha=PlayerAlpha)  # plot player positions
#                 figobjs.append(objs)
#                 if include_player_velocities:
#                     vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns]  # column header for player x positions
#                     vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns]  # column header for player y positions
#                     objs = ax.quiver(team[x_columns], team[y_columns], team[vx_columns], team[vy_columns],
#                                      color=color, scale_units='inches', scale=10., width=0.0015, headlength=5,
#                                      headwidth=3, alpha=PlayerAlpha)
#                     figobjs.append(objs)
#             # plot ball
#             objs, = ax.plot(team['ball_x'], team['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
#             figobjs.append(objs)
#             # include match time at the top
#             frame_minute = int(team['Time [s]'] / 60.)
#             frame_second = (team['Time [s]'] / 60. - frame_minute) * 60.
#             timestring = "%d:%1.2f" % (frame_minute, frame_second)
#             objs = ax.text(-2.5, field_dimen[1] / 2. + 1., timestring, fontsize=14)
#             figobjs.append(objs)
#             writer.grab_frame()
#             # Delete all axis objects (other than pitch lines) in preperation for next frame
#             for figobj in figobjs:
#                 figobj.remove()
#     print("done")
#     plt.clf()
#     plt.close(fig)
#

home_path = r'/home/non1987/Documents/football_analysis/sample-data/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv'
away_path = r'/home/non1987/Documents/football_analysis/sample-data/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv'
event_path = r'/home/non1987/Documents/football_analysis/sample-data/data/Sample_Game_2/Sample_Game_2_RawEventsData.csv'