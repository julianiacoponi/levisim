"""
Models for the x and y mechanical frequencies of a toy-parameter `MultiTrap` (ΩModel)
"""

import numpy as np

from levisim.core.units import kHz


__all__ = [
    'ΩModel',
]


class ΩModel:
    """
    Model for the x and y mechanical frequencies of a toy-parameter MultiTrap

    Kinds and their parameters (defaults in KINDS), filling the x, y labels in order::

      'pairs'                  Ω_first, Ω_intrapair_gap, Ω_interpair_gap - x,y of each bead
                               split by the intrapair gap, beads split by the interpair gap,
                               descending (copies Tania's ParametersNpair.f90)
      'equal spacing: simple'  Ω_start, Ω_gap - descending in steps of Ω_gap
      'equal spacing'          Ω_mid, Ω_span - ascending, evenly filling Ω_mid ± Ω_span/2
                               (after the 'Mechanical Mode Comb' paper)
      'degenerate'             Ω0 - every frequency equal
      'multiband'              Ω_centers, Ω_spans - the labels are shared out over the bands in
                               order (any remainder to the first bands) and each band is filled
                               evenly across center ± span/2, or sits at its center if it holds one
      'custom'                 toy_Ωjn_dict - {label: Ω} given directly (z labels too if 3 axes)

    z frequencies (3 axes) are not part of the model: MultiTrap sets Ωz = Ωx / Ωz_to_Ωx_multiplier
    for every kind except 'custom'. A model can be passed to MultiTrap as toy_params['Ω_model'],
    or given the legacy way as the kind string plus its parameters in the same toy_params dict.
    [2π·Hz]
    """
    KINDS = {
        'pairs': dict(Ω_first=300*kHz, Ω_intrapair_gap=1.5*kHz, Ω_interpair_gap=15*kHz),
        'equal spacing: simple': dict(Ω_start=300*kHz, Ω_gap=5*kHz),
        'equal spacing': dict(Ω_mid=300*kHz, Ω_span=100*kHz),
        'degenerate': dict(Ω0=300*kHz),
        'multiband': dict(Ω_centers=None, Ω_spans=None),
        'custom': dict(toy_Ωjn_dict=None),
    }

    def __init__(self, kind='pairs', **params):
        """
        Initialises an ΩModel of the given `kind`, with defaults for parameters not given
        """
        if kind not in self.KINDS:
            raise ValueError(f'Unknown Ω_model {kind!r}; expected one of {list(self.KINDS)}')
        unknown = set(params) - set(self.KINDS[kind])
        if unknown:
            raise ValueError(f'Ω_model {kind!r} does not take {sorted(unknown)}; it takes {list(self.KINDS[kind])}')
        self.kind = kind
        self.params = {**self.KINDS[kind], **params}

        if kind == 'multiband':
            if self.params['Ω_centers'] is None or self.params['Ω_spans'] is None:
                raise ValueError("Ω_model 'multiband' needs both Ω_centers and Ω_spans")
            self.params['Ω_centers'] = np.atleast_1d(np.asarray(self.params['Ω_centers'], dtype=float))
            self.params['Ω_spans'] = np.atleast_1d(np.asarray(self.params['Ω_spans'], dtype=float))
            if self.params['Ω_centers'].shape != self.params['Ω_spans'].shape:
                raise ValueError('Ω_centers and Ω_spans must have the same length')
        if kind == 'custom' and self.params['toy_Ωjn_dict'] is None:
            raise ValueError("Ω_model 'custom' needs toy_Ωjn_dict")

    def __repr__(self):
        """
        String to represent an ΩModel instance
        """
        return f'ΩModel({self.kind!r})'

    @classmethod
    def from_toy_params(cls, toy_params):
        """
        Builds the model from a toy_params dict: toy_params['Ω_model'] is either an ΩModel, or a
        kind string whose parameters are read from the same dict (other keys are ignored)
        """
        model = toy_params.get('Ω_model', 'pairs')
        if isinstance(model, cls):
            return model
        if model not in cls.KINDS:
            raise ValueError(f'Unknown Ω_model {model!r}; expected one of {list(cls.KINDS)}')
        return cls(model, **{name: toy_params[name] for name in cls.KINDS[model] if name in toy_params})

    def as_toy_params(self):
        """
        The legacy toy_params form, ``dict(Ω_model=kind, **parameters)``, for splatting into toy_params
        """
        return dict(Ω_model=self.kind, **self.params)

    def frequencies(self, labels):
        """
        {label: Ω} for every x and y label in `labels` (z labels are skipped, except for 'custom',
        which returns a copy of toy_Ωjn_dict as given)
        [2π·Hz]
        """
        p = self.params
        non_z_labels = [label for label in labels if label[0] != 'z']
        Ωjn_dict = {}

        if self.kind == 'pairs':
            Ω_value = p['Ω_first']
            for count, label in enumerate(non_z_labels):
                Ωjn_dict[label] = Ω_value
                if not count % 2:
                    # even index is the first in a pair: add the intrapair gap for the next value
                    Ω_value += p['Ω_intrapair_gap']
                else:
                    # odd index is the second in a pair: step down to the next pair
                    Ω_value -= (p['Ω_interpair_gap'] + p['Ω_intrapair_gap'])

        elif self.kind == 'equal spacing: simple':
            Ω_value = p['Ω_start']
            for label in non_z_labels:
                Ωjn_dict[label] = Ω_value
                Ω_value -= p['Ω_gap']

        elif self.kind == 'equal spacing':
            Ω_low = p['Ω_mid'] - p['Ω_span'] / 2
            # divide by one fewer than the number of things to fill the span
            Ω_gap = 0 if len(non_z_labels) == 1 else p['Ω_span'] / (len(non_z_labels) - 1)
            Ω_value = Ω_low
            for label in non_z_labels:
                Ωjn_dict[label] = Ω_value
                Ω_value += Ω_gap

        elif self.kind == 'degenerate':
            Ωjn_dict = {label: p['Ω0'] for label in non_z_labels}

        elif self.kind == 'multiband':
            num_bands = len(p['Ω_centers'])
            base, remainder = divmod(len(non_z_labels), num_bands)
            band_sizes = [base + (1 if band < remainder else 0) for band in range(num_bands)]
            remaining_labels = iter(non_z_labels)
            for center, span, size in zip(p['Ω_centers'], p['Ω_spans'], band_sizes):
                for ii in range(size):
                    label = next(remaining_labels)
                    Ωjn_dict[label] = center if size == 1 else center - span / 2 + ii * span / (size - 1)

        elif self.kind == 'custom':
            Ωjn_dict = dict(p['toy_Ωjn_dict'])

        return Ωjn_dict

    @property
    def description(self):
        """
        The model's parameters, one indented line each, as printed in MultiTrap.toy_params_info_str
        """
        p = self.params
        if self.kind == 'multiband':
            return (
                f'\n    Ω_centers = {np.array2string(p["Ω_centers"] / kHz, precision=1, separator=", ")} kHz'
                f'\n    Ω_spans = {np.array2string(p["Ω_spans"] / kHz, precision=1, separator=", ")} kHz'
            )
        if self.kind == 'custom':
            return ''
        return ''.join(f'\n    {name} = {value / kHz:.1f} kHz' for name, value in p.items())

    @property
    def info_str(self):
        """
        String describing the ΩModel instance
        """
        return f'ΩModel(\n    kind = {self.kind}{self.description}\n)'

    @property
    def info(self):
        """
        Prints information on the ΩModel instance
        """
        print(self.info_str)
