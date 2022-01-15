import numpy as np
import vapory

from abc import ABC, abstractmethod

class Geom:
    @abstractmethod
    def __call__(self):
        pass

class ElasitcaRod(Geom):
    def __init__(self, rod):
        pass

    def __call__(self):
        pass

class ElasticaCylinder(Geom):
    def __init__(self, rigid_body):
        pass

    def __call__(self):
        pass

class Sphere(Geom):
    def __init__(self, point, radius):
        pass

    def __call__(self):
        pass

class Session:
    def __init__(self, width, height):
        self.object_collection = []
        self.width = width
        self.height = height

        # Assets
        self.camera = vapory.Camera( 'location', [3,3,-5], 'look_at', [0,0,0] )
        self.light = vapory.LightSource( [2,4,-3], 'color', [1,1,1] )

    def add_rods(self, rods:list[Geom]):
        pass

    def add_rigid_body(self, body:Geom):
        pass

    def add_point(self, point:tuple[float], radius:float):
        pass

    def render(self):
        objects = [obj() for obj in self.object_collection]
        objects.append(self.light)
        scene = vapory.Scene(
                    self.camera,
                    objects=objects,
                    included=["default.inc"])
        return scene.render(width=width, height=height)
