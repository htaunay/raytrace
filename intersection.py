class Intersection:

    def __init__(self, d, norm, cube):

        self.d = d
        self.norm = norm
        self.cube = cube

    def worstCase():
        return Intersection(float("inf"), None, None)

    def __cpm__(self, other):

        if not isinstance(other, Intersection):
                raise Exception("Cannot compare Intersection with " +
                                type(other))

        return self.d - other.d
