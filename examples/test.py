from tokenize import Pointfloat
from matplotlib.pyplot import close
import shapely.geometry as geom
import math 
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import numpy as np
from util import *


l1 = geom.LineString([(0,0),(0,10)])
l2 = geom.LineString([(1,0),(1,10)])

print(l1.crosses(l2))
exit()

# p1 = geom.Point(0, 0)
# p2 = geom.Point(0, 1)
# p3 = geom.Point(0, 2)
# p4 = geom.Point(1, 2)

d_treshold = 1.5

p1 = (0,0)
p2 = (0,1)
p3 = (0,2)
p4 = (1,3)
p5 = (5,3)
p6 = (5,4)
p7 = (5,5)
p8 = (6,5)
p9 = (5,4)
p10 = (11,4)
p11 = (11,5)
p12 = (11,5.5)

points=[p1,p2,p3,p4,p5, p6,p7,p8,p9, p10, p11, p12,
 (2,3), (3,3), (4,3),
 (7,5), (8,5), (9,5)]
# segments = [[(-1,2), (-2,2), (-2,3), (-2.5, 3.5)]]
segments = []


new_points = [(2,2), (2,3), (3,3), (2,4)]



# l1 = geom.LineString(segments[0])










# assign points to segments
for point in points:
    new_segment = True
    connected_segment_index = -1
    segment_index = 0
    combine_segments = []


    for segment in segments:
        closest_dist = get_distance_from_point_to_points(point, segment)
        # print(closest_dist)
        if(closest_dist < 0.1): continue
        if(closest_dist < d_treshold):
            segment.append(point)
            # if new point already belongs to a segment, but also mathes another one, we can combine the segments
            if(not new_segment):
                if(segment_index != connected_segment_index):
                    print("We should conbine segment ", segment_index, "with segment ", connected_segment_index)
                    combine_segments.append([connected_segment_index,segment_index]) 
            connected_segment_index = segment_index
            new_segment = False
        segment_index+=1
    if(new_segment):
        segment = [point]
        segments.append(segment)

    print ("Segments", segments)


    # Combine segments
    i = 0
    print("combine segments", combine_segments)
    for c in combine_segments:
        segments[c[0] - i] += segments[c[1] -i ]
        segments.pop(c[1] - i)
        print("segments after combining", c , segments)
        i += 1





print("segments", segments)



for segment in segments:
    x_val = [x[0] for x in segment]
    y_val = [x[1] for x in segment]

    plt.plot(x_val,y_val,'o')
    # plt.show()
    plt.savefig("Myfile.png",format="png")