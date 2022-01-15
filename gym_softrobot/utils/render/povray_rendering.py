import numpy as np
import vapory

from abc import ABC, abstractmethod

import pkg_resources

class Geom:
    @abstractmethod
    def __call__(self):
        pass

class ElasticaTexture:
    pigment = Pigment( 'color', [0.45,0.39,1.0], 'transmit', 0.0 )
    texture = Texture( pigment, Finish( 'phong', 1))

class ElasitcaRod(Geom, ElasticaTexture):
    def __init__(self, rod):
        self.rod = rod

    def __call__(self):
        sphere_sweep = SphereSweep('b_spline', 4, (-2,-2,0), 0.1, [-2,2,0], 0.1, [2,-2,0],0.1,[1,1,1],0.1,
                ElasticaTexture.texture)

class ElasticaCylinder(Geom, ElasticaTexture):
    def __init__(self, body):
        self.body = body

    def __call__(self):
        sphere_sweep = SphereSweep('b_spline', 4, (-2,-2,0), 0.1, [-2,2,0], 0.1, [2,-2,0],0.1,[1,1,1],0.1,
                ElasticaTexture.texture)

class Sphere(Geom):
    def __init__(self, point, radius):
        self.point = point
        self.radius = radius
        self.sphere = Sphere( [0,1,2], 2, Texture( Pigment( 'color', [1,0,1] )))

    def __call__(self):
        return self.sphere

class Session:
    def __init__(self, width, height):
        self.object_collection = []
        self.width = width
        self.height = height

        # Assets
        self.camera = vapory.Camera( 'location', [3,3,-5], 'look_at', [0,0,0] )
        self.light = vapory.LightSource( [2,4,-3], 'color', [1,1,1] )

        self.background_path = pkg_resources.resource_string(__name__, '/'.join(['utils','render','default.inc']))

    def add_rods(self, rods):
        for rod in rods:
            self.object_collection.append(ElasticaRod(rod))

    def add_rigid_body(self, body):
        self.object_collection.append(ElasticaCylinder(body))

    def add_point(self, point:tuple[float], radius:float):
        self.object_collection.append(Sphere(body))

    def render(self):
        objects = [obj() for obj in self.object_collection]
        objects.append(self.light)
        scene = vapory.Scene(
                    self.camera,
                    objects=objects,
                    included=[self.background_path])
        return scene.render(width=width, height=height)
