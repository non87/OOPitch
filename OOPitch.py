'''
OOPitch brings Object-Oriented programming to football analytics. It is based on the most common data-analysis
libraries -- numpy (scipy), pandas and matplotlib -- and it extends the computational geometry library shapely
to account for the necessities of football analytics.

'''

from shapely_footbal import SubPitch
import numpy as np
from scipy.signal import savgol_filter
from shapely.geometry import Polygon
from shapely_footbal import Point
# We use geopandas most as a plotter. This will go away in more mature versions
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

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
            if hertz is None: raise ValueError("If positions is specified, you need to indicate Hertz as well.")
            self.create_positions(positions)
            self.calculate_velocities(hertz, smoothing=smoothing, filter_=filter_, window_length=window_length,
                                      max_speed=max_speed, **kwargs)
        else:
            self.__positions = np.nan
            self.__velocity = np.nan
            self.__smoothing = None
            self.__filter_par = None

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
        velocity_x = (self.__positions.x.iloc[1:].to_numpy() - self.__positions.x.iloc[:-1]) * hertz
        # Last point has no velocity
        velocity_x.loc[self.__positions.index[-1]] = np.nan
        velocity_y = (self.__positions.y.iloc[1:].to_numpy() - self.__positions.y.iloc[:-1]) * hertz
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
        kwargs = copy.copy(self.__filter_par)
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


class Player(ObejctOnPitch):
    '''
    Define players and all their methods/attributes
    '''

    def __init__(self, id, team, number=None, positions=None, name=None, hertz=None,
                 smoothing=True, filter_='Savitzky-Golay', window_length=7, max_speed=12, **kwargs):
        super().__init__(id, positions=positions, hertz=hertz, smoothing=smoothing,
                         filter_=filter_, window_length=window_length, max_speed=max_speed, **kwargs)
        self.__team = team
        # Set number
        if number is not None:
            self.__couple_namenumber = number
        else:
            self.__number = np.nan
        # Set name
        if name is not None:
            self.__name = name
        else:
            self.__name = np.nan
        #Without data from other players it is impossible to know if a player is a GK
        self.__is_goalkeepr = np.nan

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
        return self.__is_goalkeepr

    @GK.setter
    def GK(self, value):
        if isinstance(True, bool):
            self.__is_goalkeepr = value
        else:
            raise TypeError("The value of Player.GK is either True or False.")


class Ball(ObejctOnPitch):
    '''
    Define the ball and its property
    '''

    def __init__(self, id, positions=None, hertz=None, smoothing=True, filter_='Savitzky-Golay', window_length=7,
                 max_speed=12, **kwargs):
        super().__init__(id, positions=positions, hertz=hertz, smoothing=smoothing,
                         filter_=filter_, window_length=window_length, max_speed=max_speed, **kwargs)


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


        Returrns
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
    :param events:
    :param pitch:
    :param halves:
    :param home_name:
    :param away_name:
    :param hertz:
    '''

    def __init__(self, home_tracking, away_tracking, ball, events, pitch, halves, home_name='home', away_name='away',
                 hertz=25, colors=('red', 'blue')):
        self.__player_ids = [p.id for p in home_tracking] + [p.id for p in away_tracking]
        self.__home_tracking = {p.id: p for p in home_tracking}
        # Correct speed at the end/start of an half
        # for player in self.__home_tracking.values():
        #     print(type(player))
        #     player.correct_speed(halves, hertz)
        self.__pitch = pitch
        self.__away_tracking = {p.id: p for p in away_tracking}
        # for player in self.__away_tracking.values():
        #     player.correct_speed(halves, hertz)
        self.__ball = ball
        self.__events = events
        self.__hertz = hertz
        self.__home_name = home_name
        self.__away_name = away_name
        self.__pitch_dimension = pitch.dimension
        self.__team_names = (home_name, away_name)
        self.get_GKs()

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
        koff_frame = self.__events.loc[(self.__events['Subtype'] == 'KICK OFF') & (self.__events['Period'] == 1),
                                   'Start Frame']
        koff_frame = koff_frame.iloc[0]
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


    def plot_frame(self, frame, figax=None, team_colors=('r', 'b'), include_player_velocities=False,
                   PlayerMarkerSize=10,
                   PlayerAlpha=0.7, annotate=False):
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
                                       (as quivers). Default is False
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
        for team_name, team, color in zip(self.__team_names, [self.__home_tracking, self.__away_tracking], team_colors):
            print(f"TEAM: {team_name}")
            _ = [p for p in team.values() if isinstance(p.positions.loc[frame], Point)]
            relevant_players[team_name] = _
            # X and Y position for the home/away team
            Xs = [p.positions.loc[frame].x for p in _]
            Ys = [p.positions.loc[frame].y for p in _]
            # plot player positions
            ax.plot(Xs, Ys, color + 'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
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
            ax.plot(self.__ball.positions.loc[frame].x, self.__ball.positions.loc[frame].y, 'ko', markersize=6,
                    alpha=1.0,
                    linewidth=0, color='white')
        return fig, ax

    def save_match_clip(self, sequence, fpath, fname='clip_test', figax=None,
                        team_colors=('r', 'b'), field_dimen=(106.0, 68.0), include_player_velocities=False,
                        PlayerMarkerSize=10, PlayerAlpha=0.7):
        """ save_match_clip( hometeam, awayteam, fpath )

        Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'

        Parameters
        -----------
            fpath: directory to save the movie
            fname: movie filename. Default is 'clip_test.mp4'
            fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
            frames_per_second: frames per second to assume when generating the movie. Default is 25.
            team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
            field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
            include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
            PlayerMarkerSize: size of the individual player marlers. Default is 10
            PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7

        Returrns
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
        fname = fpath + '/' + fname + '.mp4'  # path and filename
        # create football pitch
        if figax is None:
            fig, ax = self.pitch.plot()
        else:
            fig, ax = figax
        fig.set_tight_layout(True)
        # Generate movie
        print("Generating movie...", end='')
        with writer.saving(fig, fname, 100):
            for frame in sequence:
                figobjs = []  # this is used to collect up all the axis objects so that they can be deleted after each iteration
                relevant_players = {}
                for team_name, team, color in \
                        zip(self.__team_names, [self.__home_tracking, self.__away_tracking], team_colors):
                    # Get players on the pitch
                    _ = [p for p in team.values() if isinstance(p.positions.loc[frame], Point)]
                    relevant_players[team_name] = _
                    Xs = [p.positions.loc[frame].x for p in _]
                    Ys = [p.positions.loc[frame].y for p in _]
                    # Plot players position
                    objs, = ax.plot(Xs, Ys, color + 'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
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
                # plot ball
                if isinstance(self.__ball.positions.loc[frame], Point):
                    objs, = ax.plot(self.__ball.positions.loc[frame].x, self.__ball.positions.loc[frame].y, 'ko',
                                    markersize=6, alpha=1.0, linewidth=0, color='white')
                    # objs, = ax.plot(team['ball_x'], team['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
                    figobjs.append(objs)
                # include match time at the top
                frame_minute = np.int(frame / (60 * 25))
                frame_second = np.int( np.floor((frame / (60 * 25) - frame_minute) * 60.))
                timestring = f"{frame_minute}:{frame_second}"
                objs = ax.text(-2.5, field_dimen[1] / 2. + 1., timestring, fontsize=14)
                figobjs.append(objs)
                writer.grab_frame()
                # Delete all axis objects (other than pitch lines) in preperation for next frame
                for figobj in figobjs:
                    figobj.remove()
        print("done")
        plt.clf()
        plt.close(fig)

    def plot_events(self, event_ids, figax=None, indicators=['Marker', 'Arrow'], color='r',
                    marker_style='o', alpha=0.5, annotate=False):
        """ plot_events( events )

        Plots Metrica event positions on a football pitch. event data can be a single or several rows of a data frame.
        All distances should be in meters.

        Parameters
        -----------
            events: index (or indices) for the event in the event dataframe of the match object
            fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an
                    existing figure, or None (the default) to generate a new pitch plot,
            field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
            indicators: List containing choices on how to plot the event. 'Marker' places a marker at the 'Start X/Y'
                        location of the event; 'Arrow' draws an arrow from the start to end locations. Can choose one or both.
            color: color of indicator. Default is 'r' (red)
            marker_style: Marker type used to indicate the event position. Default is 'o' (filled ircle).
            alpha: alpha of event marker. Default is 0.5
            annotate: Boolean determining whether text annotation from event data 'Type' and 'From' fields is shown on plot.
                      Default is False.

        Returrns
        -----------
             fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

        """

        if figax is None:  # create new pitch
            fig, ax = self.__pitch.plot()
        else:  # overlay on a previously generated pitch
            fig, ax = figax
        events = self.__events.loc[event_ids, :]
        for i, row in events.iterrows():
            if 'Marker' in indicators:
                # ax.plot(row['Start X'], row['Start Y'], color+marker_style, alpha=alpha )
                ax.plot(row['Start'].x, row['Start'].y, color + marker_style, alpha=alpha)
            if 'Arrow' in indicators:
                ax.annotate("", xy=row['End'].xy, xytext=row['Start'].xy,
                            alpha=alpha,
                            arrowprops=dict(alpha=alpha, width=0.5, headlength=4.0, headwidth=4.0, color=color),
                            annotation_clip=False)
            if annotate:
                text_string = row['Type'] + ': ' + row['From']
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
                         annotate=False,
                         color='k', alpha=1)

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
