'''
OOPitch brings Object-Oriented programming to football analytics. It is based on the most common data-analysis
libraries -- numpy (scipy), pandas and matplotlib -- and it extends the computational geometry library shapely
to account for the necessities of football analytics.

'''

import numpy as np
from scipy.signal import savgol_filter
from shapely.geometry import Polygon, LineString, MultiPoint, MultiPolygon
import shapely.wkt as wkt
from shapely_football import Point, SubPitch
# We use geopandas most as a plotter. This will go away in more mature versions
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import pickle as pkl
from copy import copy

meters_per_yard = 0.9144  # unit conversion from yards to meters


class ObejctOnPitch:
    '''
    Base class. Define main attributes and methods for ball and players
    '''

    def __init__(self, id, positions=None, hertz=None, smoothing=True, filter_='Savitzky-Golay', window_length=7,
                 max_speed=12, **kwargs):
        self.__id = id
        # Set positions
        if positions is not None:
            if hertz is None: raise ValueError("If positions is specified, you need to indicate hertz as well.")
            self.create_positions(positions)
            self.calculate_velocities(hertz, smoothing=smoothing, filter_=filter_, window_length=window_length,
                                      max_speed=max_speed, **kwargs)
        else:
            self.__positions = np.nan
            self.__velocity = np.nan
            self.__smoothing = None
            self.__filter_par = None
        self.__total_distance = np.nan

    # Define state methods for pickling
    def __getstate__(self):
        state = copy(self.__dict__)
        # Pickling geo-series is impossible. we get around it by saving the geo-series in a wkt format
        if self.__positions is not None:
            # Fill nas, otherwise we trigger an error when creating pos
            pos = self.__positions.copy().fillna(value=Point([-999, -999]))
            # Save positions as wkt
            pos = MultiPoint(pos.geometry.to_list())
            # We save frames for reference
            state['_ObejctOnPitch__frames'] = self.__positions.index.values
            state['_ObejctOnPitch__positions'] = pos.wkt
        else:
            state['_ObejctOnPitch__frames'] = None
        return state

    def __setstate__(self, state):
        # Load positions from wkt if necessary
        if state['_ObejctOnPitch__positions'] is not None:
            pos = wkt.loads(state['_ObejctOnPitch__positions'])
            pos = gpd.GeoSeries([Point(p) for p in pos.geoms])
            pos.loc[pos == Point([-999, -999])] = None
            pos.index = state['_ObejctOnPitch__frames']
            state['_ObejctOnPitch__positions'] = pos
        del state['_ObejctOnPitch__frames']
        self.__dict__.update(state)

    def calculate_velocities(self, hertz, smoothing=True, filter_='Savitzky-Golay', window_length=7,
                             max_speed=12, **kwargs):
        '''
        Calculate velocities of an object based on a GeoSeries of Positions.

        :param hertz: The tracking frequency per second
        :param maxspeed: Max speed after which we consider the position to be in error
        :param smoothing: Should a smoothing be applied?
        :param filter_: If a smoothing is applied, which filter should be used? Either 'Savitzky-Golay' or 'linear'
        :param kwargs: Arguments passed to the filter.
        '''
        if np.all(pd.isna(self.__positions)):
            print("No valid positions to calculate velocity")
            return None
        self.__velocity_par = {'filter': None, 'hertz': hertz, 'max_speed': max_speed}
        # Velocity is a Dataframe containing the x,y components as two columns
        velocity_x = (self.__positions.x - self.__positions.shift(-1).x) * hertz
        # Last point has no velocity
        velocity_x.loc[self.__positions.index[-1]] = np.nan
        velocity_y = (self.__positions.y - self.__positions.shift(-1).y) * hertz
        # Last point has no velocity
        velocity_y.loc[self.__positions.index[-1]] = np.nan
        velocity = pd.DataFrame(np.array([velocity_x, velocity_y], dtype=np.float32).T)
        velocity = velocity.rename(columns={0: 'x', 1: 'y'})
        velocity['speed'] = np.linalg.norm(velocity.to_numpy(), axis=1)
        self.__smoothing = False
        # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
        if max_speed is not None and max_speed > 0:
            velocity.loc[velocity['speed'] > max_speed, ['x', 'y', 'speed']] = np.nan
        if smoothing:
            if filter_ == 'Savitzky-Golay':
                self.__velocity_par['filter'] = 'Savitzky-Golay'
                self.__smoothing = savgol_filter
                # This works as default
                if 'polyorder' not in kwargs.keys():
                    kwargs['polyorder'] = 1
                # save for later
                kwargs['window_length'] = window_length
                velocity['x'] = savgol_filter(velocity['x'], **kwargs)
                velocity['y'] = savgol_filter(velocity['y'], **kwargs)
            elif filter_ == 'moving_average':
                self.__velocity_par['filter'] = 'moving_average'
                self.__smoothing = pd.DataFrame.rolling
                # save for later
                kwargs['window'] = window_length
                if 'min_periods' not in kwargs:
                    kwargs['min_periods'] = 1
                if 'center' not in kwargs:
                    kwargs['center'] = True
                velocity['x'] = velocity['x'].rolling(**kwargs).mean()
                velocity['y'] = velocity['y'].rolling(**kwargs).mean()
            else:
                raise NotImplementedError("Savitzky-Golay and moving_average are the only options for smoothing.")
            self.__filter_par = kwargs
            # After filtering, recalculate total speed
            velocity['speed'] = np.linalg.norm(velocity[['x', 'y']].to_numpy(), axis=1)
        velocity.index = self.__positions.index
        self.__velocity = velocity

    def create_positions(self, positions):
        '''
        Register and save the positions of the player during the 90 minutes. Most of the function is actually about
        creating velocity and acceleration estimates.
        :param positions: A Series containing Points
        '''
        # A geoseries actually speeds up some operations
        self.__positions = gpd.GeoSeries(positions)
        # It turns out GeoSeries transforms np.nan in None
        self.__positions.loc[self.__positions.isnull()] = np.nan

    def correct_speed(self, halves, hertz):
        '''
        The calculate_speed function does not take into accout the end of the Periods within a game. This means that
        there will be aberration in the speed calculation immediately after the second period (or extra-time) start.
        This function will correct for those aberration.

        :param halves: The frame where new halves start
        '''

        if np.all(pd.isna(self.__velocity)):
            print("No valid velocity. Have you called calculate_velocities")
            return None
        for half in halves:
            # Set the velocity of the last frame of an half to 0 for calculations
            if self.__smoothing:
                # There should be a window or window_length in the kwargs
                try:
                    window = self.__filter_par['window_length']
                    self.__velocity.loc[(half - 1), ['x', 'y', 'speed']] = 0
                except KeyError:
                    window = self.__filter_par['window']
                    self.__velocity.loc[(half - 1), ['x', 'y', 'speed']] = np.nan
                # calculate the x, y components again for those frames that are affected
                for col in ['x', 'y']:
                    before_half = np.array(range((half - 1 - window * 2), (half - 1)))
                    after_half = np.array(range(half, (half + window * 2)))
                    self.__velocity.loc[before_half, col] = (self.__positions.x.loc[before_half[1:]].to_numpy() -
                                                             self.__positions.x.loc[before_half[:-1]]) * hertz
                    self.__velocity.loc[before_half, col] = self.__smoothing(self.__velocity.loc[before_half, col],
                                                                             **self.__filter_par).mean()
                    self.__velocity.loc[after_half, col] = (self.__positions.x.loc[after_half[1:]].to_numpy() -
                                                            self.__positions.x.loc[after_half[:-1]]) * hertz
                    self.__velocity.loc[after_half, col] = self.__smoothing(self.__velocity.loc[after_half, col],
                                                                            **self.__filter_par).mean()
                # Recalculate speed
                self.__velocity.loc[(half - 1 - window * 2):(half + window * 2), 'speed'] = np.linalg.norm(
                    self.__velocity.loc[(half - 1 - window * 2):(half + window * 2), ['x', 'y']].to_numpy(), axis=1)
            # Set the velocity of the last frame of an half to nan
            self.__velocity.loc[(half - 1), ['x', 'y', 'speed']] = np.nan

    @property
    def positions(self):
        return self.__positions

    @positions.setter
    def positions(self, positions):
        self.create_positions(positions)
        kwargs = copy(self.__filter_par)
        try:
            window = kwargs['window_length']
            del kwargs['window_length']
        except KeyError:
            window = kwargs['window']
            del kwargs['window']
        self.calculate_velocities(hertz=self.__velocity_par['hertz'], smoothing=self.__smoothing,
                                  filter_=self.__velocity_par['filter'], max_speed=self.__velocity_par['max_speed'],
                                  window_length=window, **kwargs)

    @property
    def id(self):
        return (self.__id)

    @property
    def velocity(self):
        return (self.__velocity)

    @property
    def total_distance(self):
        '''
        Total distance covered in the match
        '''
        if np.isnan(self.__total_distance):
            self.__total_distance = LineString(self.positions.to_list()).length
        return(self.__total_distance)

    def plot(self, frames, pitch = None, figax = None, color='red', player_marker_size=10, player_alpha=0.7):
        '''
        Plot the positions of a player over a pitch. Return the used axis if needed
        :param frames: Which frames should be plotted
        :param pitch: A Pitch object, where the player is moving
        :param figax: A figure, axis couple. This is an alternative to the pitch
        :param color: The color of the player's marker
        :param player_marker_size: How big the marker should be
        :param player_alpha: The alpha for the plaeyer marker
        :return: The axis that has just been modified
        '''
        if pitch is None and figax is None:
            raise AttributeError("Exactly one among pitch and figax must be specified")
        if pitch is not None:
            figax = pitch.plot()
        fig, ax = figax
        ax.text(self.__positions.loc[frames[0]].x + 0.5, self.__positions.loc[frames[0]].y + 0.5, self.__id,
                          fontsize=10, color=color)
        ax = self.__positions.loc[frames].plot(ax=ax, color=color, markersize=player_marker_size, alpha=player_alpha)
        return ax

class Player(ObejctOnPitch):
    '''
    Define players and all their methods/attributes
    '''

    def __init__(self, player_id, team, number=None, positions=None, name=None, hertz=None,
                 smoothing=True, filter_='Savitzky-Golay', window_length=7, max_speed=12, **kwargs):
        super().__init__(player_id, positions=positions, hertz=hertz, smoothing=smoothing,
                         filter_=filter_, window_length=window_length, max_speed=max_speed, **kwargs)
        self.__team = team
        # Set number
        if number is not None:
            self.__number = number
        else:
            self.__number = np.nan
        # Set name
        if name is not None:
            self.__name = name
        else:
            self.__name = np.nan
        #Without data from other players it is impossible to know if a player is a GK
        self.__is_goalkeeper = np.nan

    @property
    def number(self):
        return (self.__number)

    @property
    def name(self):
        return self.__name

    @property
    def team(self):
        return self.__team

    @team.setter
    def team(self, new_team):
        self.__team = new_team

    @property
    def GK(self):
        return self.__is_goalkeeper

    @GK.setter
    def GK(self, value):
        if isinstance(True, bool):
            self.__is_goalkeeper = value
        else:
            raise TypeError("The value of Player.GK is either True or False.")

class Ball(ObejctOnPitch):
    '''
    Define the ball and its property
    '''

    def __init__(self, positions=None, hertz=None, smoothing=True, filter_='Savitzky-Golay', window_length=7,
                 max_speed=12, in_play=None, **kwargs):
        super().__init__('ball', positions=positions, hertz=hertz, smoothing=smoothing,
                         filter_=filter_, window_length=window_length, max_speed=max_speed, **kwargs)
        # DataFrame containing whether the ball is 'alive' or 'dead'
        if in_play is not None:
            assert self.positions is not None
            assert in_play.shape[0] == super().positions.shape[0]
            self.__in_play = in_play

    @property
    def in_play(self):
        return self.__in_play

    @in_play.setter
    def in_play(self, in_play):
        assert self.positions is not None
        assert in_play.shape[0] == self.positions.shape[0]
        self.__in_play = in_play

class Pitch:
    '''
    Define the Pitch where the game happens. Divide it in SubPitches for analysis sake.
    We follow the convention of making the center of the field the 0,0 point and have negative values on the
    bottom left (as a standard cartesian plane)

    field_dimension: iterable of length 2. The field dimension. Should be in meter.
    n_grid_cells_x: int. Regulates in how many SubPitch the Pitch is sub-divided into.
    '''

    def __init__(self, pitch_dimen=(106.0, 68.0), n_grid_cells_x=None):
        self.__dimension = pitch_dimen
        # Create one polygon representing the entire pitch. May be helpful for spatial operation (like, is a player
        # on the pitch? Is the ball in play?)
        self.__polygon = Polygon([(-self.__dimension[1] / 2, -self.__dimension[0] / 2),
                                  (self.__dimension[1] / 2, -self.__dimension[0] / 2),
                                  (self.__dimension[1] / 2, self.__dimension[0] / 2),
                                  (-self.__dimension[1] / 2, self.__dimension[0] / 2)])
        # Create patches for the subpitch
        if n_grid_cells_x is not None:
            self.__n_grid_cells_x = np.int(n_grid_cells_x)
            self.create_subpitch(self.__n_grid_cells_x)
        else:
            self.__n_grid_cells_x = None
            self.__n_grid_cells_y = None
            self.__subpitch = None
            self.__sub_centroids = None

    # Define state methods for pickling
    def __getstate__(self):
        state = copy(self.__dict__)
        state['_Pitch__polygon'] = state['_Pitch__polygon'].wkt
        # Pickling geo-series is impossible. we get around it by saving the geo-series in a wkt format
        if state['_Pitch__n_grid_cells_x'] is not None:
            # Save subpitches as wkt
            state['_Pitch__subpitch_inds'] = state['_Pitch__subpitch'].index.values
            subp = MultiPolygon(state['_Pitch__subpitch'].geometry.to_list())
            state['_Pitch__subpitch'] = subp.wkt
            # Save centroids as wkt
            state['_Pitch__sub_centroids_inds'] = state['_Pitch__sub_centroids'].index.values
            cents = MultiPoint( state['_Pitch__sub_centroids'].geometry.to_list() )
            state['_Pitch__sub_centroids'] = cents.wkt
        else:
            state['_Pitch__sub_centroids_inds'] = None
            state['_Pitch__subpitch_inds'] = None
        return state

    def __setstate__(self, state):
        state['_Pitch__polygon'] = wkt.loads(state['_Pitch__polygon'])
        # Load sub-pitches and their centroids from wkt if necessary
        if state['_Pitch__subpitch_inds'] is not None:
            subp = wkt.loads(state['_Pitch__subpitch'])
            subp = gpd.GeoSeries([SubPitch(p) for p in subp.geoms])
            subp.index = state['_Pitch__subpitch_inds']
            state['_Pitch__subpitch'] = subp
            cents = wkt.loads(state['_Pitch__sub_centroids'])
            cents = gpd.GeoSeries([Point(p) for p in cents.geoms])
            cents.index = state['_Pitch__sub_centroids_inds']
            state['_Pitch__sub_centroids'] = cents
        del state['_Pitch__subpitch_inds']
        del state['_Pitch__sub_centroids_inds']
        self.__dict__.update(state)


    def create_subpitch(self, n_grid_cells_x):
        # break the pitch down into a grid
        n_grid_cells_y = np.int(np.ceil((n_grid_cells_x + 1) * self.__dimension[1] / self.__dimension[0]))
        # These are the extremes of each grid cell
        xgrid = np.linspace(-self.__dimension[0] / 2., self.__dimension[0] / 2., n_grid_cells_x + 1)
        ygrid = np.linspace(-self.__dimension[1] / 2., self.__dimension[1] / 2., n_grid_cells_y + 1)
        self.__n_grid_cells_y = np.int(ygrid.shape[0] - 1)
        subpitch = []
        # navigate the grid to create subpitches
        for i in range(xgrid.shape[0] - 1):
            for j in range(ygrid.shape[0] - 1):
                # Coordinate of this subpitch
                coords = [(xgrid[i], ygrid[j]), (xgrid[i + 1], ygrid[j]),
                          (xgrid[i + 1], ygrid[j + 1]), (xgrid[i], ygrid[j + 1])]
                subpitch.append(SubPitch(coords))
        self.__subpitch = gpd.GeoSeries(subpitch)
        # Create centroids as well
        self.__sub_centroids = self.__subpitch.apply(lambda x: x.centroid)

    @property
    def dimension(self):
        return self.__dimension

    @dimension.setter
    def dimension(self, dimension):
        self.__dimension = dimension
        if self.__n_grid_cells_x is not None:
            self.create_subpitch(self.__n_grid_cells_x)

    @property
    def n_grid_cells_x(self):
        return self.__n_grid_cells_x

    @n_grid_cells_x.setter
    def n_grid_cells_x(self, n_grid_cells_x):
        if n_grid_cells_x is None:
            self.__subpitch = None
            self.__n_grid_cells_x = None
            self.__sub_centroids = None
        else:
            self.__n_grid_cells_x = np.int(n_grid_cells_x)
            self.create_subpitch(self.__n_grid_cells_x)

    @property
    def n_grid_cells_y(self):
        return self.__n_grid_cells_y

    @n_grid_cells_y.setter
    def n_grid_cells_y(self, n_grid_cells_y):
        raise NotImplementedError("At the moment, the only way to change the subpitch grid is to change n_grid_cells_x")
    #
    # @property
    # def sub_pitch(self):
    #     return(self.__subpitch)

    @property
    def sub_pitch_area(self):
        return(np.round(self.__subpitch.iloc[0].area, 3))

    def plot(self, field_color='green', linewidth=2, markersize=20, fig_ax=None, grid=False, grid_alpha=1,
             grid_col='black'):
        """
        Plots a soccer pitch. Most of this code comes from Laurie Shaw

        Parameters
        -----------
            field_color: color of field. options are {'green','white'}
            linewidth  : width of lines. default = 2
            markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20
            fig_ax: figure, axis from matplotlib. default = None
            grid: Boolean. Should the subpitch grid be plotted? Plots nothing if the pitch has no subpitch.
            grid_alpha: float in [0,1]. What alpha should the grid have
            grid_col: Color to be passed to matplotlib


        Returns
        -----------
           fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

        """
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))  # create a figure
        else:
            fig, ax = fig_ax
        # decide what color we want the field to be. Default is green, but can also choose white
        if field_color == 'green':
            ax.set_facecolor('mediumseagreen')
            lc = 'whitesmoke'  # line color
            pc = 'w'  # 'spot' colors
        elif field_color == 'white':
            lc = 'k'
            pc = 'k'
        # ALL DIMENSIONS IN m
        border_dimen = (3, 3)  # include a border arround of the field of width 3m
        half_pitch_length = self.__dimension[0] / 2.  # length of half pitch
        half_pitch_width = self.__dimension[1] / 2.  # width of half pitch
        signs = [-1, 1]
        # Soccer field dimensions typically defined in yards, so we need to convert to meters
        goal_line_width = 8 * meters_per_yard
        box_width = 20 * meters_per_yard
        box_length = 6 * meters_per_yard
        area_width = 44 * meters_per_yard
        area_length = 18 * meters_per_yard
        penalty_spot = 12 * meters_per_yard
        corner_radius = 1 * meters_per_yard
        D_length = 8 * meters_per_yard
        D_radius = 10 * meters_per_yard
        D_pos = 12 * meters_per_yard
        centre_circle_radius = 10 * meters_per_yard
        # plot half way line # center circle
        ax.plot([0, 0], [-half_pitch_width, half_pitch_width], lc, linewidth=linewidth)
        ax.scatter(0.0, 0.0, marker='o', facecolor=lc, linewidth=0, s=markersize)
        y = np.linspace(-1, 1, 50) * centre_circle_radius
        x = np.sqrt(centre_circle_radius ** 2 - y ** 2)
        ax.plot(x, y, lc, linewidth=linewidth)
        ax.plot(-x, y, lc, linewidth=linewidth)
        for s in signs:  # plots each line seperately
            # plot pitch boundary
            ax.plot([-half_pitch_length, half_pitch_length], [s * half_pitch_width, s * half_pitch_width], lc,
                    linewidth=linewidth)
            ax.plot([s * half_pitch_length, s * half_pitch_length], [-half_pitch_width, half_pitch_width], lc,
                    linewidth=linewidth)
            # goal posts & line
            ax.plot([s * half_pitch_length, s * half_pitch_length], [-goal_line_width / 2., goal_line_width / 2.],
                    pc + 's',
                    markersize=6 * markersize / 20., linewidth=linewidth)
            # 6 yard box
            ax.plot([s * half_pitch_length, s * half_pitch_length - s * box_length], [box_width / 2., box_width / 2.],
                    lc,
                    linewidth=linewidth)
            ax.plot([s * half_pitch_length, s * half_pitch_length - s * box_length], [-box_width / 2., -box_width / 2.],
                    lc,
                    linewidth=linewidth)
            ax.plot([s * half_pitch_length - s * box_length, s * half_pitch_length - s * box_length],
                    [-box_width / 2., box_width / 2.], lc, linewidth=linewidth)
            # penalty area
            ax.plot([s * half_pitch_length, s * half_pitch_length - s * area_length],
                    [area_width / 2., area_width / 2.],
                    lc, linewidth=linewidth)
            ax.plot([s * half_pitch_length, s * half_pitch_length - s * area_length],
                    [-area_width / 2., -area_width / 2.],
                    lc, linewidth=linewidth)
            ax.plot([s * half_pitch_length - s * area_length, s * half_pitch_length - s * area_length],
                    [-area_width / 2., area_width / 2.], lc, linewidth=linewidth)
            # penalty spot
            ax.scatter(s * half_pitch_length - s * penalty_spot, 0.0, marker='o', facecolor=lc, linewidth=0,
                       s=markersize)
            # corner flags
            y = np.linspace(0, 1, 50) * corner_radius
            x = np.sqrt(corner_radius ** 2 - y ** 2)
            ax.plot(s * half_pitch_length - s * x, -half_pitch_width + y, lc, linewidth=linewidth)
            ax.plot(s * half_pitch_length - s * x, half_pitch_width - y, lc, linewidth=linewidth)
            # draw the D
            y = np.linspace(-1, 1, 50) * D_length  # D_length is the chord of the circle that defines the D
            x = np.sqrt(D_radius ** 2 - y ** 2) + D_pos
            ax.plot(s * half_pitch_length - s * x, y, lc, linewidth=linewidth)

        # remove axis labels and ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        # set axis limits
        xmax = self.__dimension[0] / 2. + border_dimen[0]
        ymax = self.__dimension[1] / 2. + border_dimen[1]
        ax.set_xlim([-xmax, xmax])
        ax.set_ylim([-ymax, ymax])
        ax.set_axisbelow(True)
        ax.set_aspect('equal')
        if self.__n_grid_cells_x is not None and grid:
            self.__subpitch.plot(ax=ax, facecolor="none", edgecolor=grid_col, alpha=grid_alpha)
        return fig, ax

class Match:
    '''
    Contain all information about one match. It needs two iterables containing home and away Players, a Ball object, a
    Pitch object, an event dataset. Hertz shows the frequency (per second) of the tracking data. Other attributes will
    be specified in the future.

    :param home_tracking:
    :param away_tracking:
    :param ball:
    :param events: A DataFrame of events data. Its first column should be a frame reference -- even if no tracking data
       is passed
    :param pitch:
    :param halves:
    :param home_name:
    :param away_name:
    :param hertz:
    '''

    def __init__(self, home_tracking, away_tracking, ball, events, pitch, halves, home_name='home', away_name='away',
                 hertz=25, colors=('red', 'blue'), calculate_possession=False, possession_kwords={}):
        self.__player_ids = [p.id for p in home_tracking] + [p.id for p in away_tracking]
        self.__home_tracking = {p.id: p for p in home_tracking}
        self.__pitch = pitch
        self.__away_tracking = {p.id: p for p in away_tracking}
        self.__ball = ball
        self.__events = events
        self.__hertz = hertz
        self.__home_name = home_name
        self.__away_name = away_name
        self.__pitch_dimension = pitch.dimension
        self.__team_names = (home_name, away_name)
        self.__colors = {team:color for team, color in zip(self.__team_names, colors)}
        self.__possession_pars = possession_kwords
        # First frame new half
        self.__halves_frame = halves
        if not calculate_possession:
            self.__possession = None
        else:
            self.assign_possesion(**possession_kwords)
        # Correct speed at the end/start of an half
        # for player in self.__home_tracking.values():
        #     print(f"Correcting Velocity for Player {player.id}.")
        #     player.correct_speed(halves, hertz)
        # for player in self.away_tracking.values():
        #     print(f"Correcting Velocity for Player {player.id}.")
        #     player.correct_speed(halves, hertz)
        # print(f"Correcting Velocity for Ball.")
        # ball.correct_speed(halves, hertz)
        # TODO This will only work if we have tracking data for the players
        self.get_GKs()

    # Define state methods for pickling
    def __getstate__(self):
        state = copy(self.__dict__)
        # We need to makes sure we can pickle the players, the ball, the pitch and the events
        # Start with the ball
        state['_Match__ball'] = (state['_Match__ball'].__class__, state['_Match__ball'].__getstate__())
        # Continue with player
        state['_Match__away_tracking'] = {p_id: (p_obj.__class__, p_obj.__getstate__())
                                           for p_id, p_obj in state['_Match__away_tracking'].items()}
        state['_Match__home_tracking'] = {p_id:  (p_obj.__class__, p_obj.__getstate__())
                                           for p_id, p_obj in state['_Match__home_tracking'].items()}
        # Then the pitch
        state['_Match__pitch'] = (state['_Match__pitch'].__class__, state['_Match__pitch'].__getstate__())
        # Finally, the events
        # Check which columns are geometries
        state['_Match__event_geometry'] = {}
        events = self.__events.copy()
        state['_Match__event_orders'] = events.columns.values.copy()
        for col, dtype in zip(events.columns, events.dtypes):
            # Save the geometry columns as wkt
            if dtype == 'object' and isinstance(events.loc[~events[col].isna(), col].iloc[0], Point):
                # Pandas have a strange behavior with fillna or simple assignment
                g_col = events[col].apply(lambda x: Point([-999, -999]) if pd.isna(x) else x)
                state['_Match__event_geometry'][col] = MultiPoint(g_col.to_list()).wkt
                events = events.drop(columns=[col])
        state['_Match__events'] = events
        return state

    def __setstate__(self, state):
        # We need to rebuild the objects containing geometries
        # Start with the ball
        cls = state['_Match__ball'][0]
        ball = cls.__new__(cls)
        ball.__setstate__(state['_Match__ball'][1])
        state['_Match__ball'] = ball
        # Continue with players
        away_tracking = {}
        home_tracking = {}
        for p_id, obj in state['_Match__away_tracking'].items():
            cls = obj[0]
            p = cls.__new__(cls)
            p.__setstate__(obj[1])
            away_tracking[p_id] = p
        state['_Match__away_tracking'] = away_tracking
        for p_id, obj in state['_Match__home_tracking'].items():
            cls = obj[0]
            p = cls.__new__(cls)
            p.__setstate__(obj[1])
            home_tracking[p_id] = p
        state['_Match__home_tracking'] = home_tracking
        # Then the pitch
        cls = state['_Match__pitch'][0]
        pitch = cls.__new__(cls)
        pitch.__setstate__(state['_Match__pitch'][1])
        state['_Match__pitch'] = pitch
        # Now the events
        for col, geoms in state['_Match__event_geometry'].items():
            # Make a series
            geoms = pd.Series([Point(p) for p in wkt.loads(geoms).geoms])
            geoms.index = state['_Match__events'].index
            # Get the Nans back
            geoms[geoms == Point([-999., -999.])] = np.nan
            state['_Match__events'][col] = geoms
        state['_Match__events'] = state['_Match__events'][state['_Match__event_orders']]
        del state['_Match__event_orders'], state['_Match__event_geometry']
        self.__dict__.update(state)

    def save(self, save_path, protocol=pkl.DEFAULT_PROTOCOL):
        with open(save_path, 'wb') as fl:
            pkl.dump(self, fl, protocol=protocol)

    @property
    def pitch(self):
        return self.__pitch

    @property
    def player_ids(self):
        return self.__player_ids

    @property
    def home_tracking(self):
        return self.__home_tracking

    @property
    def away_tracking(self):
        return self.__away_tracking

    @property
    def ball(self):
        return self.__ball

    @property
    def events(self):
        return self.__events

    @property
    def hertz(self):
        return self.__hertz

    @property
    def home_team(self):
        return (self.__home_name)

    @property
    def away_team(self):
        return (self.__away_name)

    @property
    def teams(self):
        return (self.__team_names)

    @property
    def halves(self):
        return (self.__halves_frame)

    @property
    def team_colors(self):
        return(self.__colors)

    @team_colors.setter
    def team_colors(self, colors):
        '''
        Change the colors of the team
        :param colors: an iterable of length 2 containing the colors. Also a dictionary The colors must be specified in
            a way that matplotlib understads.
        '''
        if not isinstance(colors, dict):
            self.__colors = {team: color for team, color in zip(self.__team_names, colors)}
        else:
            ks = set(k for k in colors.keys())
            if ks == set(team for team in self.__team_names):
                self.__colors = colors
            else:
                raise KeyError("The key of the dictionary match the teams' name of the match object.")

    @teams.setter
    def teams(self, team_names):
        '''
        Change the team names. The names must be ordered, first home then away
        :param new_team: A 2-element iterable of containing strings
        '''
        # If we change the team names, we want to change those in every player as well
        self.__team_names = (team for team in team_names)
        for player in self.__home_tracking.values():
            player.team = self.__team_names[0]
        for player in self.__away_tracking.values():
            player.team = self.__team_names[1]

    @property
    def GKs(self):
        return({k:p.id for k, p in self.__GKs.items()})

    @property
    def attack_directions(self):
        '''
        :return: Sign of the attacking direction
        '''
        return(self.__attack_dir)

    def invert_attack_directions(self):
        '''
        Invert the attacking directions in the Match object
        '''
        for team in [self.__home_tracking, self.__away_tracking]:
            for p_id, p in team.items():
                print(f"Inverting: {p_id}")
                p._ObejctOnPitch__positions = p._ObejctOnPitch__positions.geometry.affine_transform([-1, 0, 0, -1, 0, 0])
                # Velocity is registered in floats, so we can simply multiply by -1
                p._ObejctOnPitch__velocity[['x','y']] = p._ObejctOnPitch__velocity[['x', 'y']] * (-1)
        print("Inverting ball")
        self.__ball._ObejctOnPitch__positions = \
            self.__ball._ObejctOnPitch__positions.geometry.affine_transform([-1, 0, 0, -1, 0, 0])
        self.__ball._ObejctOnPitch__velocity[['x', 'y']] = self.__ball._ObejctOnPitch__velocity[['x', 'y']] * (-1)
        for k in self.__attack_dir.keys():
            self.__attack_dir[k] = self.__attack_dir[k] * (-1)

    @property
    def possession(self):
        '''
        Possession spells. Calculate them if not already calculated
        '''
        if self.__possession is None:
            self.assign_possesion(**self.__possession_pars)
        return self.__possession

    @property
    def possession_parameters(self):
        '''
        Parameters used in the calculation of possession
        '''
        return self.__possession_pars

    @possession_parameters.setter
    def possession_parameters(self, new_pars):
        '''
        Parameters used in the calculation of possession. May contain a partial change (no need to set all parameters
        all the time)
        '''
        self.__possession_pars.update(new_pars)

    def __getitem__(self, item):
        '''
        Quick ways to get tracking data or events
        '''
        if item in self.__player_ids:
            try:
                return self.__home_tracking[item]
            except KeyError:
                return self.__away_tracking[item]
        elif item == 'ball':
            return self.__ball
        elif item == 'events':
            return self.__events
        raise KeyError(f"{item} is not `ball`, `events` or any of the players' id.")

    def get_GKs(self):
        '''
        This function infers which player is the GK based on position at kick-off. It also calculates the attack sides.
        '''
        attack_sides = {team:0 for team in self.__team_names}
        GKs = {team:np.nan for team in self.__team_names}
        koff_frame = self.__events.iloc[0].name
        # We infer attack direction based on the kickoff positions
        for team_name, team in zip(self.__team_names, [self.__home_tracking, self.__away_tracking]):
            _ = [p for p in team.values() if isinstance(p.positions.loc[koff_frame], Point)]
            att_side = np.array([p.positions[koff_frame].x for p in _])
            # This is the defense side of the team
            attack_sides[team_name] = -np.sign(att_side.mean())
            gk_pos = (attack_sides[team_name] * att_side).argmin()
            GKs[team_name] = _[gk_pos]
            for i, p in enumerate(_):
                if i != gk_pos:
                    p.GK = False
                else:
                    p.GK = True
        self.__GKs = GKs
        self.__attack_dir = attack_sides


    def plot_frame(self, frame, figax=None, include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7,
                   annotate=False):
        """ plot_frame( hometeam, awayteam )

        Plots a frame of Metrica tracking data (player positions and the ball) on a football pitch. All distances should be
        in meters.

        Parameters
        -----------
            hometeam: row (i.e. instant) of the home team tracking data frame
            awayteam: row of the away team tracking data frame
            fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an
                    existing figure, or None (the default) to generate a new pitch plot,
            team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b'
                         (blue away team)
            field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
            include_player_velocities: Boolean variable that determines whether player velocities are also plotted
                                       (26500 quivers). Default is False
            PlayerMarkerSize: size of the individual player marlers. Default is 10
            PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
            annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)

        Returns
        -----------
           fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

        """
        if figax is None:  # create new pitch
            fig, ax = self.pitch.plot()
        else:  # overlay on a previously generated pitch
            fig, ax = figax  # unpack tuple
        # plot home & away teams in order
        relevant_players = {}
        for team_name, team in zip(self.__team_names, [self.__home_tracking, self.__away_tracking]):
            print(f"TEAM: {team_name}")
            color = self.__colors[team_name]
            _ = [p for p in team.values() if isinstance(p.positions.loc[frame], Point)]
            relevant_players[team_name] = _
            # X and Y position for the home/away team
            Xs = [p.positions.loc[frame].x for p in _]
            Ys = [p.positions.loc[frame].y for p in _]
            # plot player positions
            ax.plot(Xs, Ys, color=color, marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha, linestyle="")
            if include_player_velocities:
                vx = [p.velocity.loc[frame, 'x'] for p in
                      relevant_players[team_name]]  # X component of the speed vector
                vy = [p.velocity.loc[frame, 'y'] for p in
                      relevant_players[team_name]]  # Y component of the speed vector
                ax.quiver(Xs, Ys, vx, vy, color=color,
                          scale_units='inches', scale=10., width=0.0015, headlength=5, headwidth=3,
                          alpha=PlayerAlpha)
            if annotate:
                for i, player in enumerate(relevant_players[team_name]):
                    ax.text(Xs[i] + 0.5, Ys[i] + 0.5, player.id, fontsize=10,
                            color=color)
        # plot ball
        if isinstance(self.__ball.positions.loc[frame], Point):
            ax.plot(self.__ball.positions.loc[frame].x, self.__ball.positions.loc[frame].y, markersize=6, marker='o',
                    alpha=1.0, linewidth=0, color='white', linestyle="")
        return fig, ax

    def save_match_clip(self, sequence, path, figax=None,
                        field_dimen=(106.0, 68.0), include_player_velocities=False,
                        PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False, frame_timing=False):
        """ save_match_clip( hometeam, awayteam, fpath )

        Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'

        Parameters
        -----------
            path: path to the output file
            fname: movie filename. Default is 'clip_test.mp4'
            fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
            frames_per_second: frames per second to assume when generating the movie. Default is 25.
            team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
            field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
            include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
            PlayerMarkerSize: size of the individual player marlers. Default is 10
            PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7

        Returns
        -----------
           fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

        """
        # check that indices match first
        # assert np.all(hometeam.index == awayteam.index), "Home and away team Dataframe indices must be the same"
        # in which case use home team index
        # index = self.__home_tracking[0].positions.index
        # Set figure and movie settings
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
        writer = FFMpegWriter(fps=self.__hertz, metadata=metadata)
        if path[-4:] != '.mp4': path += '.mp4'
        # fname = fpath + '/' + fname + '.mp4'  # path and filename
        # create football pitch
        if figax is None:
            fig, ax = self.pitch.plot()
        else:
            fig, ax = figax
        fig.set_tight_layout(True)
        # Generate movie
        print("Generating movie...", end='')
        with writer.saving(fig, path, 100):
            for frame in sequence:
                figobjs = []  # this is used to collect up all the axis objects so that they can be deleted after each iteration
                relevant_players = {}
                for team_name, team in zip(self.__team_names, [self.__home_tracking, self.__away_tracking]):
                    color = self.__colors[team_name]
                    # Get players on the pitch
                    _ = [p for p in team.values() if isinstance(p.positions.loc[frame], Point)]
                    relevant_players[team_name] = _
                    Xs = [p.positions.loc[frame].x for p in _]
                    Ys = [p.positions.loc[frame].y for p in _]
                    # Plot players position
                    objs, = ax.plot(Xs, Ys, color=color, marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha,
                                    linestyle="")
                    figobjs.append(objs)
                    if include_player_velocities:
                        vx = [p.velocity.loc[frame, 'x'] for p in
                              relevant_players[team_name]]  # X component of the speed vector
                        vy = [p.velocity.loc[frame, 'y'] for p in
                              relevant_players[team_name]]  # Y component of the speed vector
                        # vy_columns = -1 * np.array(vy_columns)
                        objs = ax.quiver(Xs, Ys, vx, vy, color=color,
                                         scale_units='inches', scale=10., width=0.0015, headlength=5, headwidth=3,
                                         alpha=PlayerAlpha)
                        figobjs.append(objs)
                    if annotate:
                        for i, player in enumerate(relevant_players[team_name]):
                            objs = ax.text(Xs[i] + 0.5, Ys[i] + 0.5, player.id, fontsize=10,  color=color)
                            figobjs.append(objs)

                # plot ball
                if isinstance(self.__ball.positions.loc[frame], Point):
                    objs, = ax.plot(self.__ball.positions.loc[frame].x, self.__ball.positions.loc[frame].y, marker='o',
                                    markersize=6, alpha=1.0, linewidth=0, color='white', linestyle="")
                    # objs, = ax.plot(team['ball_x'], team['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
                    figobjs.append(objs)
                # include time reference at the top
                if not frame_timing:
                    frame_minute = np.int(frame / (60 * 25))
                    frame_second = np.int( np.floor((frame / (60 * 25) - frame_minute) * 60.))
                    timestring = f"{frame_minute}:{frame_second}"
                else:
                    timestring = f"{frame}"
                objs = ax.text(-2.5, field_dimen[1] / 2. + 1., timestring, fontsize=14)
                figobjs.append(objs)
                writer.grab_frame()
                # Delete all axis objects (other than pitch lines) in preperation for next frame
                for figobj in figobjs:
                    figobj.remove()
        print("done")
        plt.clf()
        plt.close(fig)

    def plot_events(self, event_ids, figax=None, indicators=['Marker', 'Arrow'], marker_style='o', alpha=0.5,
                    annotate=False):
        """ plot_events( events )

        Plots Metrica event positions on a football pitch. event data can be a single or several rows of a data frame.
        All distances should be in meters.

        Parameters
        -----------
            event_ids: index (or indices) for the event in the event dataframe of the match object
            fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an
                    existing figure, or None (the default) to generate a new pitch plot,
            field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
            indicators: List containing choices on how to plot the event. 'Marker' places a marker at the 'Start X/Y'
                        location of the event; 'Arrow' draws an arrow from the start to end locations. Can choose one or both.
            marker_style: Marker type used to indicate the event position. Default is 'o' (filled ircle).
            alpha: alpha of event marker. Default is 0.5
            annotate: Boolean determining whether text annotation from event data 'Type' and 'From' fields is shown on plot.
                      Default is False.

        Returns
        -----------
             fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

        """

        if figax is None:  # create new pitch
            fig, ax = self.__pitch.plot()
        else:  # overlay on a previously generated pitch
            fig, ax = figax
        events = self.__events.loc[event_ids, :]
        for i, row in events.iterrows():
            color = self.__colors[row['Team'].casefold()]
            if not pd.isna(row['Start']):
                if 'Marker' in indicators:
                    ax.plot(row['Start'].x, row['Start'].y, color=color, marker=marker_style, alpha=alpha)
                if 'Arrow' in indicators:
                    if not pd.isna(row['End']):
                        ax.annotate("", xy=row['End'].xy, xytext=row['Start'].xy,
                                    alpha=alpha,
                                    arrowprops=dict(alpha=alpha, width=0.5, headlength=4.0, headwidth=4.0, color=color),
                                    annotation_clip=False)
                if annotate:
                    text_string = row['event'] + ': ' + row['From']
                    ax.text(row['Start'].x, row['Start'].y, text_string, fontsize=10, color=color)
        return fig, ax

    def plot_pitchcontrol_for_event(self, event_id, PPCF, alpha=0.7,
                                    include_player_velocities=True, annotate=False):
        """ plot_pitchcontrol_for_event( event_id, events,  tracking_home, tracking_away, PPCF )

        Plots the pitch control surface at the instant of the event given by the event_id. Player and ball positions are overlaid.

        Parameters
        -----------
            event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
            events: Dataframe containing the event data
            tracking_home: (entire) tracking DataFrame for the Home team
            tracking_away: (entire) tracking DataFrame for the Away team
            PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
            alpha: alpha (transparency) of player markers. Default is 0.7
            include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
            annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
            field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)

        NB: this function no longer requires xgrid and ygrid as an input

        Returrns
        -----------
           fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

        """

        # pick a pass at which to generate the pitch control surface

        pass_frame = self.__events.loc[event_id, 'Start Frame']
        pass_team = self.__events.loc[event_id, 'Team']

        # plot frame and event
        fig, ax = self.__pitch.plot(field_color='white')
        self.plot_frame(pass_frame, figax=(fig, ax), PlayerAlpha=alpha,
                        include_player_velocities=include_player_velocities, annotate=annotate)
        self.plot_events(self.__events.loc[event_id:event_id], figax=(fig, ax), indicators=['Marker', 'Arrow'],
                         annotate=False, alpha=1)

        # plot pitch control surface
        if pass_team == 'Home':
            cmap = 'bwr'
        else:
            cmap = 'bwr_r'
        ax.imshow(np.flipud(PPCF),
                  extent=(-self.__pitch_dimension[0] / 2., self.__pitch_dimension[0] / 2.,
                          -self.__pitch_dimension[1] / 2., self.__pitch_dimension[1] / 2.), interpolation='spline36',
                  vmin=0.0, vmax=1.0, cmap=cmap, alpha=0.5)

        return fig, ax



    def assign_possesion(self, sd=0.4, modify_event=True, min_speed_passage=0.12, max_recaliber=1.5,
                         filter_spells=True, filter_tol=4):
        '''
        A simple function to find possession spells during the game based on the position of the ball and the players.
        Substantially select the closest player to the ball as the one having possession, but also filters for dead balls and
        passess. Uses a lot of information from the event data, substantially making the assumption that any possession spell
        must leave a mark in the events. If modify_event is True, we use the possession data to input the frame of the
        events in the event data -- this on average augments the quality of the f24 Opta data.
        :param self:
        :param sd:
        :param modify_event:
        :param min_speed_passage:
        :param max_recaliber:
        :param filter_spells:
        :param filter_tol:
        :return:
        '''
        # [range(half - 5, half + 5) for half in self.__halves_frame]
        voluntary_toss = ['pass', 'clearance', 'short free-kick', 'crossed free-kick', 'throw-in', 'other free-kick',
                          'cross', 'shot', 'crossed corner', 'free-kick shot', 'keeper punch', 'goal kick']
        if filter_tol <1: filter_tol = 1

        events = self.__events.copy()
        events['recalibrated'] = False
        events['frame_safe'] = events['frame'].copy()
        # Useful for re-calibrating the events' frame

        distances = [[], []]
        ids = [[], []]

        # for i, team in enumerate([match._Match__home_tracking, match._Match__away_tracking]):
        #     print(f"Calculating possession for team {[match._Match__home_name, match._Match__away_name][i]}")
        #     for p_id, p in team.items():
        #         print(f"Calculating possession for player {p_id}")
        #         ids[i].append(p_id)
        #         temp = pd.concat([p.positions, match._Match__ball.positions], axis=1)
        #         temp.columns = ['player', 'ball']
        #         temp = temp.loc[(~pd.isna(temp['player'])) & (~pd.isna(temp['ball']))]
        #         distances[i].append(temp['ball'].distance(temp['player']))


        # Calculate the distance of every player from the ball at every frame. Takes time
        for i, team in enumerate([self.__home_tracking, self.__away_tracking]):
            print(f"Calculating possession for team {[self.__home_name, self.__away_name][i]}")
            for p_id, p in team.items():
                print(f"Calculating possession for player {p_id}")
                ids[i].append(p_id)
                temp = pd.concat([p.positions, self.__ball.positions], axis=1)
                temp.columns = ['player', 'ball']
                temp = temp.loc[(~pd.isna(temp['player'])) & (~pd.isna(temp['ball']))]
                distances[i].append(temp['ball'].distance(temp['player']))
                # distances[i].append(temp.apply(lambda x: x['ball'].distance(x['player']), axis=1))

        ids_dic = {id_: i for ids_team in ids for i, id_ in enumerate( ids_team)}
        min_team = [[], []]
        min_dists = [[], []]
        # Calculate who is the closest to the ball for each team and her distance from the ball
        for team in [0, 1]:
            # Create the team Df
            distances[team] = pd.concat(distances[team], axis=1)
            distances[team].columns = ids[team]
            distances[team] = distances[team].fillna(np.inf)
            # Get the minimum frame by frame
            min_team[team] = distances[team].idxmin(axis=1)
            # This is a way to build a fancy index
            temp = np.vstack([min_team[team].index.values, min_team[team].apply(lambda x: ids_dic[x]).to_numpy()])
            min_dists[team] = pd.Series(distances[team].to_numpy()[temp[0]-1, temp[1]])
            min_dists[team].index = min_team[team].index

        min_dists = pd.concat(min_dists, axis=1)
        min_team = pd.concat(min_team, axis=1)
        # heuristics to identify "noman's ball" and filter it out.
        # We need to be careful because the ball tracking appears to be widely inaccurate
        passes = ((min_dists.loc[:, 0] > sd * 5) & (min_dists.loc[:, 1] > sd * 5))
        # Dead ball filter if available
        try:
            alive = (self.__ball.in_play == 'Alive')
        except AttributeError:
            alive = np.zeros((self.__ball.positions.shape[0],), dtype='bool')
        # Heuristic to identify who has the ball and how sure we are
        team_possession = min_dists[(~passes) & (alive)].idxmin(axis=1)
        # Final results
        possession = pd.DataFrame({"team":min_dists[0], "player":min_dists[0], "P":min_dists[0]})
        possession.loc[(passes) | (~alive), 'team'] = np.nan
        possession.loc[(~passes) & (alive), 'team'] = team_possession
        # Use fancy indexing in np to get the right player
        possession_player = min_team[(~passes) & (alive)].to_numpy()[np.arange(team_possession.shape[0]), team_possession.to_numpy()]
        possession_player = pd.Series(possession_player, index=min_team[(~passes) & (alive)].index)
        possession.loc[(~passes) & (alive), 'player'] = possession_player
        possession.loc[(passes) | (~alive), 'player'] = np.nan
        # normal model
        n0 = norm.pdf(min_dists.loc[:, 0], scale=sd)
        n1 = norm.pdf(min_dists.loc[:, 1], scale=sd)
        possession.loc[:, 'P'] = np.max([n0,n1],axis=0)/(n0 + n1)
        # possession = possession.astype({'possession_team': 'category', 'possession_player': 'category'})
        # passes_ind = min_dists.index[passes]
        # posses_ind = min_dists.index[~passes]
        possession['spell'] = (possession.player.fillna('---') != possession.player.fillna('---').shift(1)).astype('int').cumsum()
        possession.loc[passes, 'spell'] = -99
        # Do not count -99
        spells = np.sort(possession['spell'].unique())
        spells = spells[spells != -99]
        # Sometime we split possession spell because of tracking imprecision. This loop fills some gap
        # we proceed looking behind
        for i, spell in enumerate(spells[:-1]):
            rel = possession.loc[possession['spell'] == spells[i+1], :]
            rel_minus_1 = possession.loc[possession['spell'] == spells[i], :]
            # If less than a second difference and same player we unify in the same spell
            if (rel.index.values[0] - rel_minus_1.index.values[-1] < self.__hertz) and (rel_minus_1.iloc[0,1] ==
                                                                                      rel.iloc[0,1]):
            # if (rel.index.values[0] - rel_minus_1.index.values[-1] < match.hertz) and (rel_minus_1.iloc[0,1] ==
            #                                                                           rel.iloc[0,1]):
                inds = np.arange(rel_minus_1.index.values[0], rel.index.values[-1]+1)
                possession.loc[inds, 'player'] = rel.iloc[0,1]
                possession.loc[inds, 'team'] = rel.iloc[0,0]
                possession.loc[inds, 'spell'] = rel.iloc[0,-1]
        # Now we can safely 'delete' the P
        possession.loc[possession['spell'] == -99, 'P'] = np.nan

        # Modify the pass release
        for p in possession['player'].unique():
            relevant_events = events.loc[
                (events['player'] == p) & (events['event'].isin(voluntary_toss))]
            relevant_events = relevant_events.astype({'frame':np.int})
            spells = possession.loc[possession['player'] == p, 'spell'].unique()
            for spell in spells:
                relevant = possession.loc[possession['spell'] == spell,:]
                dist_spell = distances[int(relevant.iloc[0, 0])].loc[relevant.index, p]
                going_away = dist_spell - dist_spell.shift(1)
                # If the ball is moving away from the player
                if going_away.iloc[-1] > min_speed_passage:
                    going_away.loc[going_away < min_speed_passage] = -0.1
                    going_away = np.sign(going_away)
                    going_away = going_away != going_away.shift(1)
                    going_away = going_away.cumsum()
                    # inds corresponds the frames where the spell happens and the ball is moving away quickly
                    inds = going_away[going_away == going_away.iloc[-1]].index
                    # let's check if there is a voluntary toss around the indices we found
                    if np.abs((relevant_events['frame'] - inds.values[0])).min() < (self.__hertz*(max_recaliber/2)):
                    # if np.abs((relevant_events['frame'] - inds.values[0])).min() < (match.hertz * (max_recaliber / 2)):
                        possession.loc[inds, ['player', 'team', 'P']] = np.nan
                        possession.loc[inds, ['spell']] = -99
                        # Recalibrate events if needed
                        if modify_event:
                            rel_ind = np.abs((relevant_events['frame'] - inds.min())).idxmin()
                            events.loc[rel_ind, 'recalibrated'] = True
                            events.loc[rel_ind, 'frame'] = inds.values[0]

        # Check we have not changed the time-orders of the re-calibrated events
        if any((events.loc[events.recalibrated, 'frame'] - events.loc[events.recalibrated, 'frame'].shift(1)) <= 0):
            # Find those events whose order has been changed
            un_ordered = (events.loc[events.recalibrated, 'frame'] - events.loc[
                events.recalibrated, 'frame'].shift(1)) <= 0
            un_ordered = un_ordered | ((events.loc[events.recalibrated, 'frame'] - events.loc[
                events.recalibrated, 'frame'].shift(-1)) >= 0)
            # We default back to the initial frames for the un_ordered events and set recalibrated to False
            events.loc[(events.recalibrated) & (un_ordered), 'frame'] = events.loc[(events.recalibrated) & (un_ordered), 'frame_safe']
            events.loc[(events.recalibrated) & (un_ordered), 'recalibrated'] = False

        # Re-calibrate the entire event dataset based on the re-calibrated events
        recalibrated_inds = events.loc[events.recalibrated].index.values
        for i, recalibrated_ind in enumerate(recalibrated_inds[:-1]):
            # We effectively skip the kick-offs and possibly other events at the end/beginning of periods
            ind_0 = recalibrated_inds[i]
            ind_1 = recalibrated_inds[i+1]
            # Check we are in the same period
            if events.loc[ind_0, 'period'] == events.loc[ind_1, 'period']:
                # Take the mean distance between re-calibrated frames as the frame of the events in between
                diff_0 = .5*(events.loc[ind_0, 'frame'] - events.loc[ind_0, 'frame_safe'])
                diff_1 = .5*(events.loc[ind_1, 'frame'] - events.loc[ind_1, 'frame_safe'])
                events.loc[(ind_0+1):(ind_1-1), 'frame'] = np.round_(events.loc[(ind_0+1):(ind_1-1), 'frame_safe'] + diff_0 + diff_1)
        events = events.astype({'frame':np.int})
        # Take out the 'recalibrated' and  'frame_safe' column
        events = events.iloc[:,:-2]

        if filter_spells:
            # We take note of the spells ending and beginning the halves. These may not contain events
            # match._Match__halves_frame
            periods_frame = [range(1,6), [range(half-5, half+5) for half in self.__halves_frame],
                             range(possession.index.values[-1]-5, possession.index.values[-1])]
            periods_frame = list(pd.core.common.flatten(periods_frame))
            # These are the spells we will not filter because they are very close to the end/start of periods
            period_spell = possession.loc[periods_frame, 'spell'].unique()
            # We may be repeating -99 in this vector, but that's irrelevant
            period_spell = np.append(period_spell, -99)
            filtered_out = []
            # We check if there is an event associated with the possession spell and filter for that
            for spell_id, spell in possession.groupby('spell'):
                if (spell_id not in period_spell):
                    start = spell.index.values[0]
                    end = spell.index.values[-1]
                    p_id = spell.loc[start, 'player']
                    n_relevant_rows = events.loc[(events['frame'] >= (start-filter_tol)) &
                                                     (events['frame']<= (end+filter_tol)) &
                                                     (events['player']==p_id),:].shape[0]
                    if not n_relevant_rows:
                        filtered_out.append(spell_id)

            possession.loc[possession['spell'].isin(filtered_out), ['team', 'player', 'P']] = np.nan
            possession.loc[pd.isna(possession['player']), ['spell']] = -99
            possession.loc[possession['team'] == 0, 'team'] = self.__home_name
            possession.loc[possession['team'] == 1, 'team'] = self.__away_name
            self.__possession = possession
            self.__events = events
