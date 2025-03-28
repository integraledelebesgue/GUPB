from gupb import controller
from gupb.model import arenas
from gupb.model import characters


# noinspection PyUnusedLocal
# noinspection PyMethodMayBeStatic
class Rustler(controller.Controller):
    """
    Oxidized, Blazingly Fast, Overengineered for Survival.
    """

    def __init__(self):
        pass

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Rustler)

    def __hash__(self) -> int:
        return hash(id(self))

    def decide(self,  knowledge: characters.ChampionKnowledge) -> characters.Action:
        return characters.Action.DO_NOTHING

    def praise(self, score: int) -> None:
        pass

    def reset(self, game_no: int, arena_description: arenas.ArenaDescription) -> None:
        pass

    def register(self, key) -> None:
        pass

    @property
    def name(self) -> str:
        return 'Rustler'

    @property
    def preferred_tabard(self) -> characters.Tabard:
        return characters.Tabard.RUSTLER


POTENTIAL_CONTROLLERS = [
    Rustler(),
]
