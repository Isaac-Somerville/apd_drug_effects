"""Single-cell ventricular action potential models for six species.

Each model is a published system of ODEs, re-expressed here as a subclass of
:class:`~cell_models.base.CardiacCellModel` so that they share one interface for
integration, AP feature extraction and drug block.

=======================  ============================================  ==========  ============
Class                    Source model                                  Time unit   Limit cycle
=======================  ============================================  ==========  ============
:class:`HumanModel`      ToR-ORd-dynCl, Tomek et al. 2019, 2020 (epi)  ms          20,000 beats
:class:`DogModel`        Benson et al. 2008                            ms          10,000 beats
:class:`GuineaPigModel`  Pasek et al. 2008                             **s**       1,000 beats
:class:`MouseModel`      Li & Smith 2009 (C57BL/6 WT)                  ms          3,000 beats
:class:`PigModel`        Gaur et al. 2021                              ms          2,000 beats
:class:`RabbitModel`     Mahajan et al. 2008                           ms          1,000 beats
=======================  ============================================  ==========  ============

Each model is a third-party published work, reused here under its own licence.
Full citations, source URLs and licence terms are in ``THIRD_PARTY_NOTICES.md``;
five of the six are CC BY, so that attribution is a licence condition rather than
a courtesy. Note that :class:`HumanModel` derives from GPL-3.0 source, which is
why this repository is GPL-3.0 as a whole.

"Limit cycle" is the number of beats at 1 Hz pacing after which that model's
ionic concentrations have settled; the shipped states in
``data/saved_states/baseline/`` were produced by running exactly that long.

The guinea pig model integrates in seconds, so its AP durations come back in
seconds while every other model reports milliseconds. Anything comparing
features across species must rescale it by 1000. Note that the APDs written to
``data/simulations/apd_drug_block.csv`` have already been converted, so consumers
of that file need do nothing further.
"""

from cell_models.base import CardiacCellModel
from cell_models.dog_benson import DogModel
from cell_models.guinea_pig import GuineaPigModel
from cell_models.human_torord import HumanModel
from cell_models.mouse import MouseModel
from cell_models.pig import PigModel
from cell_models.rabbit import RabbitModel

#: Species that ship with this package, in the order used for plots and tables.
SPECIES = ["Human", "Dog", "Guinea Pig", "Mouse", "Pig", "Rabbit"]

#: Maps a species name to its model class.
MODEL_CLASSES = {
    "Human": HumanModel,
    "Dog": DogModel,
    "Guinea Pig": GuineaPigModel,
    "Mouse": MouseModel,
    "Pig": PigModel,
    "Rabbit": RabbitModel,
}

#: Beats to steady state, per species. Also selects which metadata files under
#: ``data/ap_features/`` a model reads.
NUM_CYCLES_LIMIT_STATE = {
    "Human": 20_000,
    "Dog": 10_000,
    "Guinea Pig": 1_000,
    "Mouse": 3_000,
    "Pig": 2_000,
    "Rabbit": 1_000,
}


def build_model(species, **kwargs):
    """Instantiate the model for ``species``.

    Args:
        species: One of :data:`SPECIES`.
        **kwargs: Forwarded to the model class, e.g. ``drug="quinidine"``.
    """
    if species not in MODEL_CLASSES:
        raise ValueError(f"Unknown species {species!r}; expected one of {SPECIES}")
    return MODEL_CLASSES[species](**kwargs)


__all__ = [
    "CardiacCellModel",
    "HumanModel",
    "DogModel",
    "GuineaPigModel",
    "MouseModel",
    "PigModel",
    "RabbitModel",
    "SPECIES",
    "MODEL_CLASSES",
    "NUM_CYCLES_LIMIT_STATE",
    "build_model",
]
