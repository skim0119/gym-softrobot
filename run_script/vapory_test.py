import numpy as np
from vapory import *

camera = Camera( 'location', [3,3,-5], 'look_at', [0,0,0] )
light = LightSource( [2,4,-3], 'color', [1,1,1] )
sphere = Sphere( [0,1,2], 2, Texture( Pigment( 'color', [1,0,1] )))
sphere_sweep = SphereSweep('b_spline', 4, (-2,-2,0), 0.1, [-2,2,0], 0.1, [2,-2,0],0.1,[1,1,1],0.1,
    Texture( Pigment( 'color', [0.45,0.39,1.0], 'transmit', 0.0 ), Finish( 'phong', 1), 
))

scene = Scene( camera, objects= [light, sphere, sphere_sweep], included=["default.inc"])
scene.render('output.png', width=800, height=600)
