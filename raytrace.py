import json
import math
import numpy as np

from intersection import Intersection

def getAxis(camera):

    up = np.array(camera["up"])
    eye = np.array(camera["eye"])
    center = np.array(camera["center"])

    ze = (1.0 / np.linalg.norm(eye - center)) * (eye - center)
    if(np.linalg.norm(ze) != 1.0):
        raise Exception("Invalid normal Z axis: " + ze)

    xe = (1.0 / np.linalg.norm(np.cross(up, ze))) * (np.cross(up, ze))
    if(np.linalg.norm(xe) != 1.0):
        raise Exception("Invalid normal X axis: " + xe)

    ye = np.cross(ze, xe)
    if(np.linalg.norm(ye) != 1.0):
        raise Exception("Invalid normal Y axis: " + ye)

    return {"xe": xe, "ye": ye, "ze": ze}


def getRayFunc(camera, res, xe):

    eye = np.array(camera["eye"])
    near = float(camera["near"])
    fovy = float(camera["fovy"])
    wp = res["width"]
    hp = res["height"]

    hypotenuse = near / math.cos(math.radians(fovy/2.0))
    w = math.sqrt(hypotenuse * hypotenuse - near * near) * 2.0
    h = (w * hp) / wp

    # o1 = eye - near*axis["ze"] - (h/2)*axis["ye"] - (w/2)*axis["xe"]
    # pxy = o1 + w*(x/wp)*axis["xe"] + h*(y/hp)*axis["ye"]

    def ray(x, y, t):
        d = -near*axis["ze"] + \
            h*(float(y)/hp - 0.5)*axis["ye"] + \
            w*(float(x)/wp - 0.5)*axis["xe"]

        # print "d {}".format(d)
        # print "norm {}".format(np.linalg.norm(d))
        return eye + t*d

    return ray


def intersectPlane(plane, ray):

    return


def intersectCube(origin, ray, cube):

    minBound = np.array(cube["min"])
    maxBound = np.array(cube["max"])

    closestIntersect = Intersection.worstCase()

    # Front side
    p1 = np.array([minBound[0], minBound[1], minBound[2]])
    p2 = np.array([maxBound[0], minBound[1], minBound[2]])
    p3 = np.array([maxBound[0], maxBound[1], minBound[2]])
    

    return

scene = json.load(open("scene.json"))

axis = getAxis(scene["camera"])

ray = getRayFunc(scene["camera"], scene["resolution"], axis)

intersectCube(np.array(scene["camera"]["eye"]), ray, scene["objects"][1]["cube"])

# import pprint

# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(json.load(open("scene.json")))
