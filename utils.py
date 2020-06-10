from airsim import CarControls

def compute_reward_distance_to_center(distance_to_center, track_width):
    '''
    #TODO add description

    Parameters:
        

    Returns:
         
    '''

    # Calculate 3 markers that are increasingly further away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_to_center <= marker_1:
        reward = 1
    elif distance_to_center <= marker_2:
        reward = 0.5
    elif distance_to_center <= marker_3:
        reward = 0.1
    else:
        reward = -1  # likely close to off track

    return reward