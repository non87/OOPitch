'''
Extend shapely geometries to provide a simpler syntax for the operation commonly performed in football analytic
'''
from shapely.geometry import Point as pt
from shapely.geometry import Polygon as pol
import numpy as np



meters_per_yard = 0.9144  # unit conversion from yards to meters


class Point(pt):
    def __init__(self, coords):
        super().__init__(coords)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values
        Example:
          >>> xy = Point(0, 0).xy
        """
        # This returns an array.array.
        x,y = self.coords.xy
        x = list(x)
        y = list(y)
        return np.array([x[0],y[0]])


    def __add__(self, other):
        '''
        Define the addition of a point with another point or np.array
        :param other: Point or array
        :return: Depends on other. If other is a Point, returns a Point. If other is anything else, returns a np.array
        '''
        if isinstance(other, Point):
            return Point([self.x + other.x, self.y + other.y])
        elif other is None or np.isnan(other).sum():
            return np.nan
        else:
            # The strategy is to delegate all other type control to numpy.
            return self.xy + other

    def __sub__(self, other):
        '''
        Define the substraction of a point with another point or np.array
        :param other: Point or array
        :return: Depends on other. If other is a Point, returns a Point. If other is anything else, returns a np.array
        '''
        # You can do this with pd.isna(), but I don't want to introduce pd for this
        if other is None or np.isnan(other).sum():
            return np.nan
        elif isinstance(other, Point):
            return Point([self.x - other.x, self.y - other.y])
        else:
            return self.xy - other

    def __mul__(self, other):
        return(Point(self.xy * other))

    def __str__(self):
        return(f"Point at {str(self.xy)}")

class SubPitch(pol):
    '''
    We create the pitch as a Multipolygon composed of rectangles, named SubPitch. These rectangles are a
    discretization of the pitch continuity. They are used in functions that calculate something continuosly on the
    surface of the pitch
    '''
    def __init__(self, coords):
        super().__init__(coords)

    @property
    def centroid(self):
        x, y = super().centroid.xy
        return Point([list(x)[0], list(y)[0]])



