import queue
import time

import cv2
from copy import deepcopy
from tkinter import filedialog
from tkinter import Tk

from enum import Enum
import random
import numpy as np
from typing import List, Tuple, Optional, Set


class ClickHandlerMode(Enum):
    FIRST_CLICK = 0
    SECOND_CLICK = 1
    SEARCHING = 2
    DONE = 3


class CostModeType(Enum):
    HIGH_EXPENSIVE = 0
    LOW_EXPENSIVE = 1
    CHANGE_EXPENSIVE = 2
    UPHILL_EXPENSIVE = 3
    CHANGE_AND_HIGH_EXPENSIVE = 4
    STEEP_UPHILL_INCLINES_VERY_EXPENSIVE = 5


# ------------------------------------------
# TODO #1: Pick one of the options listed above (& tinker with them later to see whether your results make sense.)
#  ... or invent your own, by adding an option to the enumerator and implementing it in the cost() method.
# Note: the tests will override your choice with HIGH_EXPENSIVE, so this is just if you are running this file.

COST_MODE: CostModeType = CostModeType.HIGH_EXPENSIVE
# -------------------------------------------

# Play with this number to decide how important the altitude data is (i.e.,how willing you are to go around unpleasant
# features in the terrain.)
ALPHA = 50.0

# Cosmetic: tinker with this to change the lateral expansion of the colors within the heat map.
HEAT_MAP_SCALE = 0.3

# Which map to open?
MAP_FILENAME = "new_england height map.jpg"
# MAP_FILENAME = "volcanoes - columbia river.jpg"
# MAP_FILENAME = "USA heightmap reduced.jpg"
# MAP_FILENAME = "Small_picture.jpg"



class PathMaker:
    def __init__(self, filename=None):
        """
        # *********************
        # NOTE: This section of code would have allowed you to pick the file with a GUI dialog, but is blocking
        #       the cv.imshow() method from making windows, so I've deactivated it..
        root = Tk()
        print("Showing file dialog. Make sure it isn't hiding!")
        root.withdraw()
        map_filename: str = filedialog.askopenfilename(message="Find the heightmap.")
        print (map_filename)
        root.destroy()
        # *********************
        """
        if filename is None:
            filename = MAP_FILENAME
        self.original_map: np.ndarray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # load the image file
        self.drawing_map: np.ndarray = cv2.cvtColor(self.original_map, cv2.COLOR_GRAY2BGR)  # convert it into color.
        print(f"Map loaded. {self.drawing_map.shape =}")

        self.click_mode = ClickHandlerMode.FIRST_CLICK
        self.waiting_for_click: bool = True
        self.start_point_x_y: Optional[Tuple[int, int]] = None
        self.start_point_r_c: Optional[Tuple[int, int]] = None
        self.end_point_x_y: Optional[Tuple[int, int]] = None
        self.end_point_r_c: Optional[Tuple[int, int]] = None

        # The "record," split into a pointer for each point to where the path came from and the "g" total cost to arrive
        #      at this point.
        self.previous_point: Optional[np.ndarray] = None
        self.best_g: Optional[np.ndarray] = None

        self.show_map()

    def get_height_at_rc(self, point: Tuple[int, int]) -> float:
        """

        Note: I've written this convenience method to illustrate the conversion
        from 0-255 to 0.0 to 1.0.
        :param point: a location in (r,c) format.
        :return: a value from 0.0 - 1.0 representing the brightness (height) at
        this point.
        """
        return self.original_map[point[0]][point[1]] / 255.0

    def get_neighbors_of(self, pt: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        :param pt: an in-bounds point as (r,c)
        :return: a list of in-bounds points to investigate along with a weight corresponding
        to the distance from pt to this point.
        """
        neighbors: List[Tuple[Tuple[int, int], float]] = []
        for row_offset in range(-1, 2):
            for col_offset in range(-1, 2):
                if row_offset == 0 and col_offset == 0:
                    continue  # we don't want to include the original point.

                if pt[0] + row_offset < 0 or pt[0] + row_offset >= self.original_map.shape[0] or \
                        pt[1] + col_offset < 0 or pt[1] + col_offset >= self.original_map.shape[1]:
                    continue  # don't include points that are out of bounds.

                # if i or j is zero, then this is a N,S,E,W path, and should have weight 1.
                if row_offset * col_offset == 0:
                    neighbors.append(((pt[0] + row_offset, pt[1] + col_offset), 1.0))

                # .... otherwise, this is a diagonal move, and we want weight √2.
                else:
                    neighbors.append(((pt[0] + row_offset, pt[1] + col_offset), 1.414))
        return neighbors

    def cost(self, point1: Tuple[int, int], point2: Tuple[int, int], dist: float = 1.0) -> float:
        """
        gives a numerical value that indicates the cost of the single step from point1 to point2, based on both lateral
        and altitude information.
        :param point1: the (r,c) location of a pixel on the map
        :param point2: the (r,c)location of an adjacent pixel on the map
        :param dist: the distance between the two pixels, probably 1.00 or 1.41.
        :return: the cost function - how expensive is it to move from pixel1 to pixel2? This must be a POSITIVE number.
        """
        result: float = 0.0

        h1: float = self.original_map[point1[0], point1[1]] / 256.0
        h2: float = self.original_map[point2[0], point2[1]] / 256.0
        if COST_MODE == CostModeType.CHANGE_EXPENSIVE:
            result = dist + ALPHA * abs(h2 - h1)

        elif COST_MODE == CostModeType.LOW_EXPENSIVE:
            result = dist + ALPHA * (256 - h2)  # low elevations are expensive

        elif COST_MODE == CostModeType.HIGH_EXPENSIVE:
            result = dist + ALPHA * h2  # high elevations are expensive

        elif COST_MODE == CostModeType.UPHILL_EXPENSIVE:
            # elevation changes are expensive, uphill twice as much as downhill.
            if h2 > h1:
                result = dist + ALPHA * (h2 - h1)
            else:
                result = dist + ALPHA * ((h1 - h2) / 2)

        elif COST_MODE == CostModeType.CHANGE_AND_HIGH_EXPENSIVE:
            result = dist + ALPHA * (abs(h2 - h1) + h2)  # elevation changes and high elevations are expensive.

        elif COST_MODE == CostModeType.STEEP_UPHILL_INCLINES_VERY_EXPENSIVE:
            if h2 > h1:
                result = dist + ALPHA * 256 * (h2 - h1) ** 2
            else:
                result = dist

        assert (result > 0, "The cost function must always produce positive numbers.")
        assert (result >= dist, "The cost function must be at least as big as the horizontal displacement to the "
                                "finish.")
        return result

    def heuristic(self, point: Tuple[int, int]) -> float:
        """
        gives a numerical value that is NO MORE than the least possible cost of the path from this point to
             self.end_point.
        :param point: a location in (r,c) coordinates
        :return: a consistent numerical value that is less than or equal to the least possible cost of the path from
                    this point to the end point.
        """
        heuristic_result = 0
        # ------------------------------------------
        # TODO #3: You should write this method
        #  I recommend using the euclidean or manhattan distance from point to self.end_point_r_c.


        # ------------------------------------------
        return heuristic_result

    def perform_search(self) -> Optional[Tuple[int, int]]:
        """
        Uses the A* algorithm to try to detect the optimal path from self.start_point to self.end_point.
        :return: either the self.end_point, if we found a path, or None - if we didn't.
        """
        start: Tuple[int, int] = self.start_point_r_c
        end: Tuple[int, int] = self.end_point_r_c

        self.draw_start_point()
        self.draw_end_point()

        # together, self.previous_point and self.best_g make up the "record" from the video.
        # self.previous point is an array of the same size as the map, holding (-1,-1) values that indicate
        #    the best path to here..
        self.previous_point: Optional[np.ndarray] = np.ones((self.original_map.shape[0], self.original_map.shape[1], 2),
                                                            dtype=int)
        self.previous_point *= -1  # start all points at (-1,-1)

        # self.best_g is an array the same size as the map indicating the least expense to get from the start point
        #    to this point; the starting value is 9 x 10**9 for all points, which is very large.
        self.best_g: Optional[np.ndarray] = np.ones((self.original_map.shape[0], self.original_map.shape[1]),
                                                    dtype=float)
        self.best_g *= 9E9 #start all points at 9 * 10**9.
        count = 0
        result = None  # hopefully, you'll improve on this.

        # ------------------------------------------
        # TODO #4: You need to write the rest of this method.
        # Consider what you need to do before you loop through the search cycle. You'll need to create a "frontier"
        # variable.

        # loop while there are still elements in frontier.
        while True:  # replace "True" with a condition about the frontier, or exit the loop when you get there.

            pt = (0, 0)  # I've put this here so the next part doesn't freak out, but pt should be the point you've
            # just popped off the frontier.





            #  optional... every few (1000?) loops, draw the path that leads to pt and update a "heat map" that shows
            #  what self.best_g looks like. You might find this interesting to observe what is going on as the computer
            #  works.
            count += 1
            if count % 1000 == 0:
                self.display_path(pt,(random.randint(64, 255), random.randint(64, 255), random.randint(64, 255)))
                self.show_map()
                self.draw_heat_map()
                self.draw_elevation_graph(pt)  # maybe...
                cv2.waitKey(1)
        # ------------------------------------------
        # print(f"{count=}")

        return result

    #  =============================================================================== DRAWING METHODS
    def set_color_at_rc(self, color: Tuple[int, int, int], point: Tuple[int, int]):
        """
        changes the color of drawing_map at the given (r,c) point to the color
        (b,g,r) in range 0-255.
        Note: you will still need to imshow the display map for this change to be
        seen by the user.
        :param color: a 0-255 color in format (b, g, r)
        :param point:  a point in format (r,c)
        :return: None
        """
        self.drawing_map[point[0]][point[1]] = color

    def draw_start_point(self):
        """
        draws a marker on the self.drawing_map at location self.start_point_x_y.
        Note that the cv2 drawing functions work in (x,y) coords, not (r,c)!
        :return: None
        """
        cv2.circle(self.drawing_map, center=self.start_point_x_y, radius=10,
                   color=(0, 192, 0), thickness=1)
        cv2.line(self.drawing_map, (self.start_point_x_y[0] - 10, self.start_point_x_y[1]),
                 (self.start_point_x_y[0] + 10, self.start_point_x_y[1]),
                 color=(0, 192, 0), thickness=1)
        cv2.line(self.drawing_map, (self.start_point_x_y[0], self.start_point_x_y[1] - 10),
                 (self.start_point_x_y[0], self.start_point_x_y[1] + 10),
                 color=(0, 192, 0), thickness=1)

    def draw_end_point(self):
        """
        draws a marker on the self.drawing_map at location self.end_point_x_y.
        Note that the cv2 drawing functions work in (x,y) coordinates, not (r,c)!
        :return: None
        """
        cv2.circle(self.drawing_map, center=self.end_point_x_y, radius=10,
                   color=(0, 0, 192), thickness=1)
        cv2.line(self.drawing_map, (self.end_point_x_y[0] - 10, self.end_point_x_y[1]),
                 (self.end_point_x_y[0] + 10, self.end_point_x_y[1]),
                 color=(0, 0, 192), thickness=1)
        cv2.line(self.drawing_map, (self.end_point_x_y[0], self.end_point_x_y[1] - 10),
                 (self.end_point_x_y[0], self.end_point_x_y[1] + 10),
                 color=(0, 0, 192), thickness=1)

    def display_path(self, path_terminator: Tuple[int, int] = None, color: Tuple[int, int, int] = (0, 192, 255)):
        """
        Draws the path tracing backward from path_terminator.

        Modifies the existing self.drawing_map graphics variable.
        :param path_terminator: the last position in a sequence of positions that make the path we are drawing;
                                or None, if no path can be found.
        :param color: the color of the line to draw, in BGR 0-255 format.
        :return: None
        """
        if path_terminator is None:
            print("No path found.")
            return
        pt: Tuple[int, int] = deepcopy(path_terminator)
        # -----------------------------------------
        # TODO #2: You should write this method
        #       hint: make use of self.set_color_at_rc(color, point)


        # -----------------------------------------

    def show_map(self):
        """
        causes the self.drawing_map to display/update.
        :return:
        """
        cv2.imshow("Map", self.drawing_map)
        cv2.moveWindow("Map", 0, 0)

    def draw_heat_map(self):
        """
        An optional debugging tool that might be helpful - it draws a visual representation of self.best_g. in a window
        called "Heat".
        I wouldn't do this EVERY frame... it will slow the search down A LOT. But now and then it might be helpful.
        :return: None
        """

        # A fancy, FAST trick to generate the original image if best_g is the starting value of 9E9, otherwise
        #   full red, no blue and a green proportional to the mod of the best_g and 180.
        # the trick works because a value like "self.best_g[:, :] < 9E9" is a boolean that
        #   resolves to 1 if true and 0 if false. The colons correspond to an inherent loop over all the indices.

        heat_map = cv2.cvtColor(self.original_map, cv2.COLOR_GRAY2BGR)

        heat_map[:, :, 0] = 0 + \
                            (self.best_g[:, :] >= 9E9) * heat_map[:, :, 0]
        heat_map[:, :, 1] = ((self.best_g[:, :] * HEAT_MAP_SCALE) % 255) * (self.best_g[:, :] < 9E9) + \
                            (self.best_g[:, :] >= 9E9) * heat_map[:, :, 1]
        heat_map[:, :, 2] = 255 * (self.best_g[:, :] < 9E9) + \
                            (self.best_g[:, :] >= 9E9) * heat_map[:, :, 2]

        cv2.imshow("Heat", heat_map)
        cv2.moveWindow("Heat", heat_map.shape[1], 0)

    def draw_elevation_graph(self, path_terminator: Tuple[int, int]):
        """
        displays a graph of the elevation along the path that ends at the given point. A possible debugging tool,
        or useful to see what the final path looks like, "from the side."
        :param path_terminator: the end point of a path.
        :return: None
        """
        cv2.imshow("Elevation Graph", self.generate_elevation_path_graph(path_terminator))
        cv2.moveWindow("Elevation Graph", 0, self.drawing_map.shape[0] + 50)

    def generate_elevation_path_graph(self, path_terminator: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        generates the elevation graph used by draw_elevation_graph. Probably only used internally.
        :param path_terminator: the end point of a path.
        :return: an ndarray graphic of the graph to display.
        """
        if path_terminator is None:
            print("No path found.")
            return None
        num_steps: int = 0
        pt: Tuple[int, int] = deepcopy(path_terminator)
        while pt[0] != -1 and pt[1] != -1:
            num_steps += 1
            pt = self.previous_point[pt[0], pt[1]]

        result = np.zeros((256, num_steps + 1, 3), dtype=float)
        x: int = 0
        pt: Tuple[int, int] = deepcopy(path_terminator)
        last_h: int = -1
        while pt[0] != -1 and pt[1] != -1:
            x += 1
            h: int = self.original_map[pt[0], pt[1]]
            if last_h > -1:
                cv2.line(result, (num_steps - x, 255 - h), (num_steps - x + 1, 255 - last_h), (0, 255, 0), thickness=1)
            last_h = h
            pt = self.previous_point[pt[0], pt[1]]

        return result

    #  ============================================================================ MOUSE AND GAME LOOP METHODS
    def start_process(self):
        """
        this is essentially our game loop - it sets up the mouse listener,
        and then enters an infinite loop where it waits for the user to select
        the two cities before it performs a search and displays the result.
        :return:
        """
        # tell the program that if anybody does anything mouse-related in the "Map" window, call self.handleClick().
        cv2.setMouseCallback("Map", self.handle_click)
        self.reset()
        while True:
            # kill time while we wait on the mouse clicks for the ends.
            while self.click_mode != ClickHandlerMode.SEARCHING:
                cv2.waitKey(1)
            # Ok, now we're ready to do the search.
            path = self.perform_search()

            # Display the results of our search.
            self.drawing_map: np.ndarray = cv2.cvtColor(self.original_map, cv2.COLOR_GRAY2BGR)
            self.draw_start_point()
            self.draw_end_point()
            self.display_path(path)
            self.show_map()
            self.draw_elevation_graph(path)

            # TODO: consider the following. No action is required.
            # Optional: if you would like to save a copy of the graphic that results,
            # you can say:
            #  cv2.imsave("pickAFilename.png",self.drawing_map).

            # Wait for the user to click again to restart the process.
            print("Click on screen once to start again.")
            self.click_mode = ClickHandlerMode.DONE
            while self.click_mode == ClickHandlerMode.DONE:
                cv2.waitKey(1)

    def reset(self):
        """
        restores the displayed image to the originally loaded graphic and prepares to wait for first point
        :return:
        """
        self.click_mode = ClickHandlerMode.FIRST_CLICK
        self.drawing_map = cv2.cvtColor(self.original_map, cv2.COLOR_GRAY2BGR)
        self.show_map()

    def handle_click(self, event, x, y, unused_flags, unused_param):
        """
        this method gets called whenever the user moves or clicks or does
        anything mouse-related while the mouse is in the "Map" window.
        In this particular case, it will only do stuff if the mouse is being
        released. What it does depends on the self.click_mode enumerated variable.
        :param event: what kind of mouse event was this?
        :param x:
        :param y:
        :param unused_flags: I suspect this will be info about modifier keys (e.g. shift) Needed parameter because the
                               event handler is calling this method, but we're going to ignore what it sends us.
        :param unused_param: additional info from cv2... probably unused. Needed parameter because the
                               event handler is calling this method, but we're going to ignore what it sends us.
        :return: None
        """
        if event != cv2.EVENT_LBUTTONUP:  # only worry about when the mouse is released inside this window.
            return

        if self.click_mode == ClickHandlerMode.FIRST_CLICK:
            # we were waiting for the user to click on the first city, and she has just done so.
            # identify which city was selected, set the self.first_city_id variable
            # and display the selected city on screen.
            self.start_point_x_y = (x, y)
            self.start_point_r_c = (y, x)
            self.draw_start_point()

            # update the screen with these changes.
            self.show_map()

            # now prepare to receive the second city.
            self.click_mode = ClickHandlerMode.SECOND_CLICK
            return

        elif self.click_mode == ClickHandlerMode.SECOND_CLICK:
            # we were waiting for the user to click on the second city, and she has just done so.
            # identify which city was selected, set the self.second_city_id variable
            # and display the selected city on screen.
            self.end_point_x_y = (x, y)
            self.end_point_r_c = (y, x)
            self.draw_end_point()
            # update the screen with these changes
            self.show_map()
            self.click_mode = ClickHandlerMode.SEARCHING
            return

        # elif self.click_mode == ClickHandlerMode.SEARCHING:
        #     # advance to the next step
        #     self.waiting_for_click = False
        #     return

        elif self.click_mode == ClickHandlerMode.DONE:
            # we just finished the search, and user has clicked, so let's start over
            self.reset()
            return

    def wait_for_click(self):
        """
        makes the program freeze until the user releases the mouse in the window.
        :return: None
        """
        self.waiting_for_click = True
        while self.waiting_for_click:
            cv2.waitKey(1)


# -----------------------------------------------------------------------------------------------------------------
# This is what the program will actually do.... like Java's main() method. (Since it's not inside the class
#    declaration.) The "if" statement means this will only happen if this file is the one that is run.
if __name__ == "__main__":
    path_maker = PathMaker()
    path_maker.start_process()

    # traditionally, this will wait indefinitely until the user presses a key and
    # then close the windows and quit. The loop in this program will make it so that
    # it never really gets here, but it's a good habit.
    cv2.waitKey()
    cv2.destroyAllWindows()
