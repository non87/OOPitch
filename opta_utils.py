import numpy as np
import pandas as pd
from pendulum import from_format
import xml.etree.ElementTree as ET

'''
This file contains variables useful to parse Opta f24 data

Without a clear guide about the collection of Opta data, I identified 3 main sources of potential error and noise
in the conversion from f24 to SPADL.

1. The event category "49" (ball recovery) does not match clearly to neither intercept nor tackle. This is a problem 
 with the SPADL codes as well. There are the only two recovery categories in SPADL (intercept, tackle), but it is clear 
 that other events may lead to ball recovery. Eg within the SPADL coding, if a 'Take on' fail, the team loses possession, 
 but this is not associated necessarily with a defensive event (like a tackle). As for event category "49", this source 
 of noise entails that it is impossible to know exactly who has the ball outside the event times in a SPADL coding. THis
 is not a problem in the f24 file though, so we are losing information.

2. There is some noise in the x, y coordinates of events, especially passes. For example, the x_end, y_end coordinates
 of passages are encoded within the passage event (qualifier_id="140", qualifier_id="141"), but they do not generally 
 coincide with the x,y coordinate of the next event if when they should -- for example, with intercept events or
 firt-touch plays. The same goes for goal-keepers' coordinates
 
3. It is unclear what criteria Opta uses to assign the "outcome" attribute of events. We use this attribute to assign
 the outcome in the SPADL coding, but it is not clear that the "outcome" assignment follows the same criteria indicated
 in the paper "Actions Speak Louder than Goals", Table 4. For example, an offside passage may have an "outcome"==1 in 
 the F24 file.
'''

# A list of all events that have any relevance for the SPADL format
relevant = [7, 10, 11, 53, 54, 41, 52, 1, 61, 12, 8, 3, 1, 2, 4, 17, 13, 14, 15, 16, 30, 32, 37]

# Events that change possession with alive ball; we do not include 49 Ball recovery
win_poss = [7, 8]

# Hertz of tracab tracking
opta_hertz = 25

# # Relevant qualifiers for shooting body parts
# shoot_bp = {15:"Head", 72:"Left foot", 20:"Right foot", 21:"Other"}
#
# # Relevant qualifiers for passing body parts
# pass_bp = {3: "Head", 168: "Head"}

#TODO: type_id="49" (ball recovery) seems another way to conquer possession again. Perhaps it is associated to situation
# where there is no clear possession for either team. Either merge this with intercept/tackle or create a category.
# The category is sometime associated with intercept or tackles, but sometimes stands alone!

def _update_spadl(spadl, new_row):
    '''
    Utility to update the spadl list of list within the f24_2_SPDAL function
    '''

    for i, el in enumerate(new_row):
        spadl[i].append(el)
    return spadl

def check_and_update(previous_event, event_type, spadl, new_row, timestamp, team, player, x, y, quals):
    '''
    Given an event and some information on the next event, provide the spadl+ row for the event

    :param previous_event: the event type_if of the event we will encode in spadl+
    :param event_type: the next event type_id
    :param spadl: spadl so far
    :param new_row: spadl+ information on the event we will encode
    :param timestamp: UTC timestamp of the next event
    :param team: team of the next event
    :param player: player of the next event
    :param x: x-coordinate of the next event
    :param y: y-coordinate of the next event
    :param quals: qualifiers id for the next event
    '''
    # If there is a 'recovery' in less than a second from an intercept/tackle from the same player, we register
    # no event
    if event_type == 49:
        if (previous_event in [8, 7]) and (timestamp - new_row[0] < 1) and (player == new_row[2]):
            event_type = -99
    # If we have an intercept/involuntary touch after a pass, we change the pass ending to get a little more consistency
    elif (event_type in [8, 61]) and (previous_event == 1):
        new_row[5] = x
        new_row[6] = y
        if team != new_row[2]:
            new_row[5] = 100 - x
            new_row[6] = 100 - y
    # Similar consistency trick, but for saved shots
    # I believe you always have an event-type 15 following an event-type 10
    elif (event_type in [15, 61]) and (previous_event == 10):
        new_row[5] = x
        new_row[6] = y
        if team != new_row[2]:
            new_row[5] = 100 - x
            new_row[6] = 100 - y
    # Put card special event for fouls.
    elif event_type == 17:
        # We may be missing some due to the condition that the *previous* event is the foul
        if previous_event == 4:
            assert((31 in quals) or (32 in quals) or (33 in quals))
            # first yellow card
            if 31 in quals:
                new_row[11] = 'yellow card (first)'
            # second yellow card
            elif 32 in quals:
                new_row[11] = 'yellow card (second)'
            # red card
            else:
                new_row[11] = 'red card'
        # We don't register cards as independent event
        event_type = -99
    # For events that are modified by next event simply register them
    spadl = _update_spadl(spadl, new_row)

    return(event_type, spadl)

def parse_passages(event, event_type = None, qual_leaf = None, quals = None):
    '''
    From a pass event (and possibly its parsed type and qualifiers) returns spadl information
    '''
    # TODO: Assumes that any passage (type=1) becomes offside passages (type=2) if cause offside.
    #  Make sure this is how it works
    if event_type is None:
        event_type = int(event.attrib['type_id'])
    assert event_type in [1,2]
    if (qual_leaf is None) or (quals is None):
        quals = [(qual, int(qual.attrib['qualifier_id'])) for qual in event.findall("Q")]
        # Separate ids from leaves
        qual_leaf = {q[1]: q[0] for q in quals}
        quals = [q[1] for q in quals]
    special = np.nan if event_type == 1 else 'Offside'
    assert (140 in quals) and (141 in quals)
    # Get the target as registred by the Opta data
    x_end = float(qual_leaf[140].attrib['value'])
    y_end = float(qual_leaf[141].attrib['value'])
    # Kloppy assigns the outcome of a pass differently (if there is a target x, y)
    # But it looks like there is *always* a target x, y
    outcome = 'Success' if int(event.attrib['outcome']) else 'Failure'
    # Just make sure we count offsides as failure
    outcome = 'Failure' if event_type == 2 else outcome
    if 3 not in quals and 168 not in quals:
        body_part = 'either feet'
    else:
        body_part = 'head'
    # Assumes free-kick (qual=5), corners (qual=6), throw-ins (qual=107) are mutually exclusive
    # Throw-in
    if 107 in quals:
        body_part = np.nan
        spadl_event = 'throw-in'
    # Free kick
    elif 5 in quals:
        # crossed free kick
        if 2 in quals:
            spadl_event = 'crossed free-kick'
        # Passes free kicks
        else:
            assert 212 in quals
            # short free kick
            if float(qual_leaf[212].attrib['value']) < 10:
                spadl_event = 'short free-kick'
            # Other free kick
            else:
                spadl_event = 'other free-kick'
    # Corner
    elif 6 in quals:
        # Crossed corners
        if 2 in quals:
            spadl_event = 'crossed corner'
        else:
            assert 212 in quals
            # short corner
            if float(qual_leaf[212].attrib['value']) < 15:
                spadl_event = 'short corner'
            # Other corner
            else:
                spadl_event = 'other corner'
    # Open play cross
    elif 2 in quals:
        spadl_event = 'cross'
    # Just normal pass
    else:
        spadl_event = 'pass'
    # Return all new info
    return spadl_event, x_end, y_end, outcome, body_part, special

def parse_shots(event, event_type = None, qual_leaf = None, quals = None):
    '''
    From a shot event (and possibly its parsed type and qualifiers) returns spadl information
    '''
    if event_type is None:
        event_type = int(event.attrib['type_id'])
    assert event_type in [13, 14, 15, 16]
    if (qual_leaf is None) or (quals is None):
        quals = [(qual, int(qual.attrib['qualifier_id'])) for qual in event.findall("Q")]
        # Separate ids from leaves
        qual_leaf = {q[1]: q[0] for q in quals}
        quals = [q[1] for q in quals]
    # In case the ball went out
    # This will cause a small imprecision with those shoots that end in throw-ins
    # Not many of these, though.
    if event_type in [13, 14, 16]:
        x_end = 100
    # For blocked shot, we will get the x coordinate from next event
    else:
        x_end = -1
    y_end = float(qual_leaf[102].attrib['value'])
    # The body part qualifier should be mutually exclusive
    assert (sum([15 in quals, 20 in quals, 21 in quals, 72 in quals])) == 1
    if 20 in quals:
        body_part = 'right foot'
    elif 72 in quals:
        body_part = 'left foot'
    elif 15 in quals:
        body_part = 'head'
    else:
        body_part = 'other'
    outcome = "Failure" if event_type != 16 else "Success"
    # Hopefully, the own goal qualifier gets added to all kind of attempt (set pieces included)
    # Own goal special
    if 28 in quals:
        y_end = 100 - y_end
        special = "Own Goal"
    else:
        special = np.nan
    # free kick shot
    if 26 in quals:
        spadl_event = "free-kick shot"
    elif 9 in quals:
        spadl_event = 'penalty'
    else:
        spadl_event = 'shot'
    # Return all new info
    return spadl_event, x_end, y_end, outcome, body_part, special

def parse_keeper(event, event_type = None, qual_leaf = None, quals = None):
    if event_type is None:
        event_type = int(event.attrib['type_id'])
    assert event_type in [10, 11, 41, 52, 53, 54]
    if (qual_leaf is None) or (quals is None):
        quals = [(qual, int(qual.attrib['qualifier_id'])) for qual in event.findall("Q")]
        # Separate ids from leaves
        qual_leaf = {q[1]: q[0] for q in quals}
        quals = [q[1] for q in quals]
    # For goalkeeper events all these fields are not relevant
    x_end, y_end, body_part, special = np.nan, np.nan, np.nan, np.nan
    outcome = "Success" if event_type != 54 else "Failure"
    # 94 signal a shoot blocked by an out-field player
    if (event_type == 10):
        if 94 not in quals:
            spadl_event = 'keeper save'
        else:
            # SPADL +
            spadl_event = 'shot blocked'
    # I Map smother on claiming cross.
    elif (event_type in [11, 53, 54]):
        spadl_event = 'keeper claim'
    # Pick up
    elif (event_type == 52):
        spadl_event = "keeper pick-up"
    # Punch
    elif (event_type == 41):
        spadl_event = "keeper punch"
    # Return all new info
    return spadl_event, x_end, y_end, outcome, body_part, special

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
            # if event_type in [30, 32, 37]: print("HEY")
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
        frames = np.int_(np.round(timestamps * opta_hertz)) + 1
    else:
        timestamps = spadl['frame'] - start_time_frame
        frames = np.int_(np.round(timestamps * opta_hertz)) + 1
    spadl['frame'] = frames
    # We need to subtract the interval length from second period (and further) frames
    for i, boundary in enumerate(period_boundaries):
        if (i % 2 == 0) and (i > 0):
            interval_lenght_in_frame = np.int_(np.round((period_boundaries[i] - period_boundaries[i-1])*opta_hertz))
            period = (i / 2) + 1
            spadl.loc[spadl['period']== period, 'frame'] = spadl.loc[spadl['period']== period, 'frame'] - interval_lenght_in_frame

    return(spadl)

def read_f24(f24_file):
    '''
    Read an f24 xml opta file and return a pandas DataFrame
    :param f24_file: the path of the xml file
    :return: a pandas DataFrame in the SPADL+ format
    '''
    f24 = ET.parse(f24_file).getroot()
    f24 = f24.find("Game")
    spadl = f24_2_SPDAL(f24)
    return(spadl)

# read_f24("odense_own_goal.xml")