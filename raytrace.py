import json
import math
import pygame
import numpy as np

from intersection import Intersection


def getAxis(camera):

    up = np.array(camera["up"])
    eye = np.array(camera["eye"])
    center = np.array(camera["center"])

    ze = ((eye - center) / np.linalg.norm(eye - center))
    if(not np.allclose(np.linalg.norm(ze), 1.0)):
        raise Exception("Invalid normal Z axis: " + str(ze))

    xe = ((np.cross(up, ze)) / np.linalg.norm(np.cross(up, ze)))
    if(not np.allclose(np.linalg.norm(xe), 1.0)):
        raise Exception("Invalid normal X axis: " + str(xe))

    ye = np.cross(ze, xe)
    if(not np.allclose(np.linalg.norm(ye), 1.0)):
        raise Exception("Invalid normal Y axis: " + str(ye))

    return {"xe": xe, "ye": ye, "ze": ze}


def getRayFunc(camera, res, xe):

    near = float(camera["near"])
    fovy = float(camera["fovy"])
    wp = res["width"]
    hp = res["height"]

    hypotenuse = near / math.cos(math.radians(fovy/2.0))
    w = math.sqrt(hypotenuse * hypotenuse - near * near) * 2.0
    h = (w * hp) / wp

    def ray(x, y, t=1):
        d = -near*axis["ze"] + \
            h*(float(y)/hp - 0.5)*axis["ye"] + \
            w*(float(x)/wp - 0.5)*axis["xe"]

        d /= np.linalg.norm(d)
        if(not np.allclose(np.linalg.norm(d), 1.0)):
            raise Exception("Invalid normal D direction: " + str(d))

        return t*d

    return ray


def intersectTriangle(p1, p2, p3, origin, ray):

    n = np.cross((p2 - p1), (p3 - p2))
    n /= np.linalg.norm(n)
    if(not np.allclose(np.linalg.norm(n), 1.0)):
        raise Exception("Invalid plane normal: " + str(n))

    ti = np.dot((p1 - origin), n) / np.dot(ray, n)
    if(ti <= 0):
        return None

    pi = origin + ti * ray

    a1 = np.dot(n, np.cross((p3 - p2), (pi - p2))) / 2.0
    a2 = np.dot(n, np.cross((p1 - p3), (pi - p3))) / 2.0
    a3 = np.dot(n, np.cross((p2 - p1), (pi - p1))) / 2.0
    ar = a1 + a2 + a3

    l1 = a1 / ar
    l2 = a2 / ar
    l3 = a3 / ar

    if(l1 >= 0 and l1 <= 1 and l2 >= 0 and l2 <= 1 and l3 >= 0 and l3 <= 1):
        return Intersection(np.linalg.norm(ti), n, pi, None)

    return None


def intersectCube(origin, ray, cube):

    minBound = np.array(cube["min"])
    maxBound = np.array(cube["max"])

    closestIntersect = Intersection.worstCase()

    # Front side
    p1 = np.array([minBound[0], minBound[1], minBound[2]])
    p2 = np.array([maxBound[0], minBound[1], minBound[2]])
    p3 = np.array([maxBound[0], maxBound[1], minBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    p1 = np.array([minBound[0], minBound[1], minBound[2]])
    p2 = np.array([maxBound[0], maxBound[1], minBound[2]])
    p3 = np.array([minBound[0], maxBound[1], minBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    # Back side
    p1 = np.array([minBound[0], minBound[1], maxBound[2]])
    p2 = np.array([minBound[0], maxBound[1], maxBound[2]])
    p3 = np.array([maxBound[0], maxBound[1], maxBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    p1 = np.array([minBound[0], minBound[1], maxBound[2]])
    p2 = np.array([minBound[0], maxBound[1], maxBound[2]])
    p3 = np.array([maxBound[0], maxBound[1], maxBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    # Right side
    p1 = np.array([maxBound[0], minBound[1], minBound[2]])
    p2 = np.array([maxBound[0], maxBound[1], minBound[2]])
    p3 = np.array([maxBound[0], maxBound[1], maxBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    p1 = np.array([maxBound[0], minBound[1], minBound[2]])
    p2 = np.array([maxBound[0], maxBound[1], maxBound[2]])
    p3 = np.array([maxBound[0], minBound[1], maxBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    # Left side
    p1 = np.array([minBound[0], minBound[1], minBound[2]])
    p2 = np.array([minBound[0], maxBound[1], minBound[2]])
    p3 = np.array([minBound[0], maxBound[1], maxBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    p1 = np.array([minBound[0], minBound[1], minBound[2]])
    p2 = np.array([minBound[0], maxBound[1], maxBound[2]])
    p3 = np.array([minBound[0], maxBound[1], minBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection
    
    # Up side
    p1 = np.array([minBound[0], maxBound[1], minBound[2]])
    p2 = np.array([maxBound[0], maxBound[1], maxBound[2]])
    p3 = np.array([maxBound[0], maxBound[1], minBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    p1 = np.array([minBound[0], maxBound[1], minBound[2]])
    p2 = np.array([minBound[0], maxBound[1], maxBound[2]])
    p3 = np.array([maxBound[0], maxBound[1], maxBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    # Down side
    p1 = np.array([minBound[0], minBound[1], minBound[2]])
    p2 = np.array([maxBound[0], minBound[1], minBound[2]])
    p3 = np.array([maxBound[0], minBound[1], maxBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    p1 = np.array([minBound[0], minBound[1], minBound[2]])
    p2 = np.array([maxBound[0], minBound[1], maxBound[2]])
    p3 = np.array([minBound[0], minBound[1], maxBound[2]])
    intersection = intersectTriangle(p1, p2, p3, origin, ray)
    if(intersection != None and intersection < closestIntersect):
        intersection.obj = cube
        closestIntersect = intersection

    if(closestIntersect.distance < 1000):
        return closestIntersect

    return None


def intersectSphere(origin, ray, sphere):

    center = np.array(sphere["center"])
    radius = float(sphere["radius"])

    closestIntersect = Intersection.worstCase()

    a = np.dot(ray, ray)
    b = np.dot(2*ray, (origin - center))
    c = np.dot((origin - center), (origin - center)) - (radius * radius)

    delta = b*b - 4*a*c
    if(delta > 0):

        t1 = (-b - math.sqrt(delta)) / (2*a)
        t2 = (-b + math.sqrt(delta)) / (2*a)
        ti = min(t1, t2)

        if(ti > 0):

            pi = origin + ray*ti
            n = (pi - center)
            n /= np.linalg.norm(n)

            return Intersection(ti, n, pi, sphere)

    return None


def intersectObjs(origin, ray, scene):

    icube = intersectCube(origin, ray, scene["objects"][1]["cube"])
    isphere = intersectSphere(origin, ray, scene["objects"][0]["sphere"])
   
    inter = None
    if(icube):
        inter = icube

    if(isphere):
        if(not inter or isphere < icube):
            inter = isphere

    return inter
        
def colour(obj, lights, camera, intersection):

    eye = np.array(camera["eye"])
    Iamb = np.array(lights["ambient"])
    Ipoint = np.array(lights["point"]["intensity"])
    pointPos = np.array(lights["point"]["pos"])

    diffuse = np.array(obj["diffuse"])
    specReflection = np.array(obj["specular"])[:3]
    specCoefficient = np.array(obj["specular"])[3]

    Camb = Iamb * diffuse

    l = pointPos - intersection.point
    l /= np.linalg.norm(l)
    Cdiff = max(np.dot(l, intersection.norm), 0) * (Ipoint * diffuse)

    v = eye - intersection.point
    v /= np.linalg.norm(v)

    r = l - 2*(np.dot(l, intersection.norm))*intersection.norm
    Cspec = math.pow(max(np.dot(-r, v), 0), specCoefficient) * \
            (Ipoint * specReflection)

    return Camb + Cdiff + Cspec


def ambient(obj, lights):

    Iamb = np.array(lights["ambient"])
    diffuse = np.array(obj["diffuse"])
    return Iamb * diffuse


scene = json.load(open("scene.json"))

axis = getAxis(scene["camera"])

rayFunc = getRayFunc(scene["camera"], scene["resolution"], axis)

pygame.init()
screen = pygame.display.set_mode([500,500])
done = False
clock = pygame.time.Clock()

lightPos = np.array(scene["lights"]["point"]["pos"])

while True:
    
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            break

    clock.tick(1)
    if(done):
        continue

    screen.fill((255,255,255))

    for i in range(0,500):
        for j in range(0,500):
            ray = rayFunc(j, i)

            inter = intersectObjs(np.array(scene["camera"]["eye"]), ray, scene)
            if(inter):
                # print "inter " + str(inter.point) 

                lightRay = lightPos - inter.point
                lightRay /= np.linalg.norm(lightRay)
                lightInter = intersectObjs(inter.point + lightRay*0.01, lightRay, scene)

                if(not lightInter):
                    c = colour(inter.obj, scene["lights"], scene["camera"], inter)

                    c *= 255
                    c[0] = c[0] if c[0] <= 255 else 255
                    c[1] = c[1] if c[1] <= 255 else 255
                    c[2] = c[2] if c[2] <= 255 else 255
                    screen.set_at((j,500 - i), c)
                else:
                    c = ambient(inter.obj, scene["lights"])
                    c *= 255
                    screen.set_at((j,500 - i), c)

            else:

                screen.set_at((j,500 - i), (230,230,230))

   
        pygame.display.update()
    done = True
    
pygame.quit();
