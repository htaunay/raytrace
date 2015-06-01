class Intersection:

    def __init__(self, distance, norm, point, obj):

        self.distance = distance
        self.norm = norm
        self.point = point
        self.obj = obj

    @staticmethod
    def worstCase():
        return Intersection(float("inf"), None, None, None)

    def __lt__(self, other):

        if not isinstance(other, Intersection):
                raise Exception("Cannot compare Intersection with " +
                                type(other))

        return self.distance < other.distance
