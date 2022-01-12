from elastica.callback_functions import CallBackBaseClass


class RigidCylinderCallBack(CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())

        return

class RodCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["dilatation"].append(system.dilatation.copy())
            self.callback_params["voronoi_dilatation"].append(system.voronoi_dilatation.copy())
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["director"].append(system.director_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["omega"].append(system.omega_collection.copy())
            self.callback_params["sigma"].append(system.sigma.copy())
            self.callback_params["kappa"].append(system.kappa.copy())
        return

class ExternalLoadCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params['external_force'].append(system.external_forces.copy())
            self.callback_params['external_couple'].append(system.external_torques.copy())
        return

class CylinderCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["radius"].append(system.radius)
            self.callback_params["height"].append(system.length)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params['director'].append(system.director_collection.copy())
        return

class SphereCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["radius"].append(system.radius)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["director"].append(system.director_collection.copy())
        return

class AlgorithmCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["director"].append(system.director_collection.copy())
            self.callback_params["sigma"].append(system.sigma.copy())
            self.callback_params["kappa"].append(system.kappa.copy())
        return

class OthersCallBack:
    def __init__(self, step_skip: int, callback_params: dict):
        self.current_step = 0
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, time, **kwargs):
        if self.current_step % self.every == 0:
            self.callback_func(time, **kwargs)
        self.current_step += 1

    def callback_func(self, time, **kwargs):
        self.callback_params['time'].append(time)
        for key, value in kwargs.items():
            # make sure value is a numpy array
            self.callback_params[key].append(value.copy())

    def save_data(self, **kwargs):

        import pickle

        print("Saving additional data to simulation_others.pickle file...", end='\r')

        with open("simulation_others.pickle", "wb") as others_file:
            data = dict(
                time_series_data=self.callback_params,
                **kwargs
            )
            pickle.dump(data, others_file)

        print("Saving additional data to simulation_others.pickle file... Done!")
