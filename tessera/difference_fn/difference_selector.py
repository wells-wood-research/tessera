import inspect
import typing as t

from tessera.difference_fn import angle_difference, shape_difference, sequence_difference
from tessera.difference_fn.angle_difference import AngleDifferenceStrategy
from tessera.difference_fn.shape_difference import ShapeDifferenceStrategy
from tessera.difference_fn.sequence_difference import SequenceDifferenceStrategy


def _register_strategies() -> t.Tuple[
    t.Dict[str, t.Dict[str, t.Type]], t.Dict[str, str]
]:
    """
    Scans modules for subclasses of difference strategies and registers them.

    Returns
    -------
    strategies : dict
        A dictionary of the form {strategy_type: {strategy_name: strategy_class}}

    """
    strategy_type_to_name: t.Dict[str, t.Dict[str, t.Type]] = {
        "angle": {},
        "shape": {},
        "sequence": {},
    }
    strategy_name_to_type: t.Dict[str, str] = {}

    module_map = {
        "angle": angle_difference,
        "shape": shape_difference,
        "sequence": sequence_difference,
    }

    for strategy_type, module in module_map.items():
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(
                obj,
                (
                    AngleDifferenceStrategy,
                    ShapeDifferenceStrategy,
                    SequenceDifferenceStrategy,
                ),
            ) and obj not in {
                AngleDifferenceStrategy,
                ShapeDifferenceStrategy,
                SequenceDifferenceStrategy,
            }:

                # Remove "differencestrategy" from difference name
                name = name.lower().replace("differencestrategy", "")
                strategy_type_to_name[strategy_type][name] = obj
                strategy_name_to_type[name] = strategy_type

    return strategy_type_to_name, strategy_name_to_type


STRATEGY_TYPE_TO_NAME, NAME_TO_STRATEGY_TYPE = _register_strategies()


def difference_function_selector(
    difference_name: str,
) -> t.Union[
    AngleDifferenceStrategy, ShapeDifferenceStrategy, SequenceDifferenceStrategy
]:
    """
    Factory function to create the appropriate difference strategy based on the user input

    Parameters
    ----------
    difference_name: str
        The name of the difference strategy to create

    Returns
    -------
    DifferenceStrategy
        The difference strategy object
    """
    difference_name = difference_name.lower()
    difference_type = NAME_TO_STRATEGY_TYPE.get(difference_name)

    if not difference_type:
        raise ValueError(
            f"Difference {difference_name} not recognized. Please choose from {NAME_TO_STRATEGY_TYPE.keys()}"
        )

    if difference_type == "shape":
        raise NotImplementedError(
            "Shape difference is deprecated due to silent failures in PyMol and BioPython. It will be removed in future versions."
        )
    strategy_cls = STRATEGY_TYPE_TO_NAME.get(difference_type, {}).get(difference_name)
    if not strategy_cls:
        raise ValueError(f"Difference '{difference_name}' not found.")

    return strategy_cls()
