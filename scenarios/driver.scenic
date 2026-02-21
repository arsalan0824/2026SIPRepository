param map = localPath('../CARLA/Town01.xodr')
param carla_map = 'Town03'

model scenic.simulators.metadrive.model

param time_step = 1.0/10
param verifaiSamplerType = 'halton'
param render = 1
param use2DMap = True

param extra_cars = 5

import numpy as np
TERMINATE_TIME = 40 / globalParameters.time_step

def get_nearest_centerline(obj):
	min_dist = np.inf
	for lane in network.lanes:
		dist = distance to lane
		if dist < min_dist:
			min_dist = dist
			centerline = lane.centerline
	return centerline

behavior WalkBehavior(speed=1.0):
	while True:
		take SetThrottleAction(0.1)
		wait

param select_road = VerifaiOptions([*network.roads])
param distractor_road = VerifaiOptions([*network.roads])

param select_lane = VerifaiOptions([*network.lanes])
param distractor_lane = VerifaiOptions([*network.lanes])

start = Uniform(*globalParameters.select_lane.centerline.points)
start2 = Uniform(*globalParameters.distractor_lane.centerline.points)

start = (start[0] @ start[1])
start2 = (start2[0] @ start2[1])

ego = new Car on start, facing roadDirection, with observation 0, with cte 0 

distractor = new Car on start2, with behavior DriveAvoidingCollisions(target_speed=10, avoidance_threshold=12)

random_lane1 = Uniform(*network.lanes)
random_point1 = Uniform(*random_lane1.centerline.points)
random_pos1 = (random_point1[0] @ random_point1[1])
intersection_car1 = new Car on random_pos1, facing roadDirection, 
	with behavior DriveAvoidingCollisions(target_speed=Range(5, 15), avoidance_threshold=10)

random_lane2 = Uniform(*network.lanes)
random_point2 = Uniform(*random_lane2.centerline.points)
random_pos2 = (random_point2[0] @ random_point2[1])
intersection_car2 = new Car on random_pos2, facing roadDirection, 
	with behavior DriveAvoidingCollisions(target_speed=Range(5, 15), avoidance_threshold=10)

random_lane3 = Uniform(*network.lanes)
random_point3 = Uniform(*random_lane3.centerline.points)
random_pos3 = (random_point3[0] @ random_point3[1])
intersection_car3 = new Car on random_pos3, facing roadDirection, 
	with behavior DriveAvoidingCollisions(target_speed=Range(5, 15), avoidance_threshold=10)

random_lane4 = Uniform(*network.lanes)
random_point4 = Uniform(*random_lane4.centerline.points)
random_pos4 = (random_point4[0] @ random_point4[1])
intersection_car4 = new Car on random_pos4, facing roadDirection, 
	with behavior DriveAvoidingCollisions(target_speed=Range(5, 15), avoidance_threshold=10)

ped_lane1 = Uniform(*network.lanes)
ped_point1 = Uniform(*ped_lane1.centerline.points)
ped_base_pos1 = (ped_point1[0] @ ped_point1[1])
pedestrian1 = new Car at ped_base_pos1 offset by Range(3, 5) @ 0,
	facing Range(0, 360) deg,
	with behavior WalkBehavior(speed=Range(0.5, 2.0))

ped_lane2 = Uniform(*network.lanes)
ped_point2 = Uniform(*ped_lane2.centerline.points)
ped_base_pos2 = (ped_point2[0] @ ped_point2[1])
pedestrian2 = new Car at ped_base_pos2 offset by Range(3, 5) @ 0,
	facing Range(0, 360) deg,
	with behavior WalkBehavior(speed=Range(0.5, 2.0))

ped_lane3 = Uniform(*network.lanes)
ped_point3 = Uniform(*ped_lane3.centerline.points)
ped_base_pos3 = (ped_point3[0] @ ped_point3[1])
pedestrian3 = new Car at ped_base_pos3 offset by Range(3, 5) @ 0,
	facing Range(0, 360) deg,
	with behavior WalkBehavior(speed=Range(0.5, 2.0))

ped_lane4 = Uniform(*network.lanes)
ped_point4 = Uniform(*ped_lane4.centerline.points)
ped_base_pos4 = (ped_point4[0] @ ped_point4[1])
pedestrian4 = new Car at ped_base_pos4 offset by Range(3, 5) @ 0,
	facing Range(0, 360) deg,
	with behavior WalkBehavior(speed=Range(0.5, 2.0))

ped_lane5 = Uniform(*network.lanes)
ped_point5 = Uniform(*ped_lane5.centerline.points)
ped_base_pos5 = (ped_point5[0] @ ped_point5[1])
pedestrian5 = new Car at ped_base_pos5 offset by Range(3, 5) @ 0,
	facing Range(0, 360) deg,
	with behavior WalkBehavior(speed=Range(0.5, 2.0))

monitor DrivingReward(obj):
	while True:
		ego.previous_coordinates = obj.position
		lane = obj._lane

		if lane:
			centerline = lane.centerline	
		else:
			centerline = get_nearest_centerline(obj)

		if obj._lane:
			ego.lane_heading = lane._defaultHeadingAt(ego.position)
			orientation_error = np.abs(ego.heading - lane._defaultHeadingAt(ego.position))
			ego.orientation_error = orientation_error

			if orientation_error > .05:
				orientation_error = max(-orientation_error, -3)
			else:
				orientation_error = 1

		nearest_line_points = centerline.nearestSegmentTo(obj.position)
		nearest_line_segment = PolylineRegion(nearest_line_points)
		
		cte = min(abs(distance to nearest_line_segment), 1)
		if cte < .2: 
			cte = 0
		speed_reward = max(0.5 * ego.speed, 2)

		dist_reward = distance to ego.previous_coordinates

		reward = -cte + speed_reward + orientation_error + dist_reward

		ego.reward = reward

		wait

require monitor DrivingReward(ego)