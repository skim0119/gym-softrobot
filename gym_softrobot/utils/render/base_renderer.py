from typing import Iterable
from abc import ABC, abstractmethod

from gym_softrobot.config import RendererType

from elastica.rod import RodBase


class BaseRenderer(ABC):
    """
    Renderer should contains the methods below.
    """

    @property
    @abstractmethod
    def type(self) -> RendererType:
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass

class BaseElasticaRendererSession(ABC):
    """
    Elastica-renderer should contains the methods below
    """

    @abstractmethod
    def add_rod(self, rod: RodBase):
        pass

    def add_rods(self, rods: Iterable[RodBase]):
        # Batch collector
        for rod in rods:
            self.add_rod(rod)

    @abstractmethod
    def add_rigid_body(self):
        pass

    @abstractmethod
    def add_point(self):
        pass
    
