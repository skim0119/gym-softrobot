import numpy as np
import vapory

from abc import ABC, abstractmethod

import pkg_resources

"""
For postion, change coordinate s.t.
http://www.povray.org/documentation/view/3.6.1/15/
For some reason, POVray use left-hand system
(x,y,z) -> (x,z,y)
"""

class Geom:
    @abstractmethod
    def __call__(self):
        pass

class ElasticaTexture:
    pigment = vapory.Pigment( 'color', [0.45,0.39,1.0], 'transmit', 0.0 )
    texture = vapory.Texture( pigment, vapory.Finish( 'phong', 1))

class ElasticaRod(Geom, ElasticaTexture):
    def __init__(self, rod):
        self.rod = rod

    def __call__(self):
        n_data = self.rod.n_elems+1
        pos_rad_pair = []
        pos = self.rod.position_collection 
        rad = self.rod.radius
        rad = np.concatenate([rad[0:1], 0.5*(rad[:-1]+rad[1:]), rad[-1:]])
        for i in range(n_data):
            x,z,y = pos[:,i] # transformation
            pos_rad_pair.append([x,y,z])
            pos_rad_pair.append(rad[i])

        return vapory.SphereSweep(
            'b_spline',
            n_data,
            *pos_rad_pair,
            ElasticaRod.texture
        )

class ElasticaCylinder(Geom):
    pigment = vapory.Pigment( 'color', [0.35,0.29,1.0], 'transmit', 0.0 )
    texture = vapory.Texture( pigment, vapory.Finish( 'phong', 1))

    def __init__(self, body):
        self.body = body

    def __call__(self):
        rad = self.body.radius[0]
        length = self.body.length
        tangent = self.body.director_collection[2,:,0]
        position1 = self.body.position_collection[:,0]
        position2 = position1 + length * tangent
        position1 = [position1[0], position1[2], position1[1]]
        position2 = [position2[0], position2[2], position2[1]]
        return vapory.Cylinder(
            position1,
            position2,
            rad, 
            ElasticaCylinder.texture
        )

class Sphere(Geom):
    pigment = vapory.Pigment( 'color', [1,0,1], 'transmit', 0.0 )
    texture = vapory.Texture( pigment, vapory.Finish( 'phong', 1))

    def __init__(self, loc, radius):
        x,z,y = loc
        self.sphere = vapory.Sphere(
            [x,y,z],
            radius,
            Sphere.texture
        )

    def __call__(self):
        return self.sphere

class Session:
    def __init__(self, width, height):
        self.object_collection = []
        self.width = width
        self.height = height

        # Assets
        self.camera = vapory.Camera( 'location', [1.7,0.7,-1.2], 'look_at', [0.5,0,1] )
        self.light = vapory.LightSource( [2,4,-3], 'color', [1,1,1] )

        #self.background_path = "default.inc"
        self.background_path = pkg_resources.resource_filename(
            __name__,
            'default.inc'
        )

    def add_rods(self, rods):
        for rod in rods:
            self.object_collection.append(ElasticaRod(rod))

    def add_rigid_body(self, body):
        self.object_collection.append(ElasticaCylinder(body))

    def add_point(self, loc:list, radius:float):
        self.object_collection.append(Sphere(loc, radius))

    def render(self):
        objects = [obj() for obj in self.object_collection]
        objects.append(self.light)
        scene = vapory.Scene(
                    self.camera,
                    objects=objects,
                    included=[self.background_path])
        return scene.render(width=self.width, height=self.height)

    def close(self):
        self.object_collection.clear()
