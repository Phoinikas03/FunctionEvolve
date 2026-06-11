from typing import Optional, Any

import json
from pathlib import Path

import numpy as np
import h5py
import datasets
from huggingface_hub import snapshot_download

from .dataclasses import Equation, Problem

import warnings

REPO_ID = "nnheui/llm-srbench"


def _download(repo_id):
    return snapshot_download(repo_id=repo_id, repo_type="dataset")


def _resolve_dataset_dir(hf_local_path: Optional[str]) -> Path:
    """Use a local snapshot directory if provided; otherwise download from the Hub."""
    if hf_local_path:
        p = Path(hf_local_path).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"HF local dataset path is not a directory: {p}")
        return p
    return Path(_download(repo_id=REPO_ID))


def _load_hf_split(hf_local_path: Optional[str], split_name: str):
    """Load a named split from the Hub or from a local dataset directory."""
    if hf_local_path:
        return datasets.load_dataset(str(Path(hf_local_path).expanduser().resolve()))[split_name]
    return datasets.load_dataset(REPO_ID)[split_name]


class TransformedFeynmanDataModule:
    def __init__(self, hf_local_path: Optional[str] = None):
        self._hf_local_path = hf_local_path
        self._dataset_dir = None
        self._dataset_identifier = "lsr_transform"

    def setup(self):
        self._dataset_dir = _resolve_dataset_dir(self._hf_local_path)
        ds = _load_hf_split(self._hf_local_path, "lsr_transform")
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        self.problems = []
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {
                    k: v[...].astype(np.float64)
                    for k, v in sample_file[f'/lsr_transform/{e["name"]}'].items()
                }
                self.problems.append(
                    Problem(
                        dataset_identifier=self._dataset_identifier,
                        equation_idx=e["name"],
                        gt_equation=Equation(
                            symbols=e["symbols"],
                            symbol_descs=e["symbol_descs"],
                            symbol_properties=e["symbol_properties"],
                            expression=e["expression"],
                        ),
                        samples=samples,
                    )
                )
        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}

    @property
    def name(self):
        return "LSR_Transform"


class SynProblem(Problem):
    @property
    def train_samples(self):
        return self.samples["train_data"]

    @property
    def test_samples(self):
        return self.samples["id_test_data"]

    @property
    def ood_test_samples(self):
        return self.samples["ood_test_data"]


class BaseSynthDataModule:
    def __init__(
        self,
        dataset_identifier,
        short_dataset_identifier,
        root,
        default_symbols=None,
        default_symbol_descs=None,
        hf_local_path: Optional[str] = None,
    ):
        self._dataset_dir = Path(root)
        self._dataset_identifier = dataset_identifier
        self._short_dataset_identifier = short_dataset_identifier
        self._default_symbols = default_symbols
        self._default_symbol_descs = default_symbol_descs
        self._hf_local_path = hf_local_path

    def setup(self):
        self._dataset_dir = _resolve_dataset_dir(self._hf_local_path)
        ds = _load_hf_split(self._hf_local_path, f"lsr_synth_{self._dataset_identifier}")
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        self.problems = []
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {
                    k: v[...].astype(np.float64)
                    for k, v in sample_file[
                        f'/lsr_synth/{self._dataset_identifier}/{e["name"]}'
                    ].items()
                }
                self.problems.append(
                    Problem(
                        dataset_identifier=self._dataset_identifier,
                        equation_idx=e["name"],
                        gt_equation=Equation(
                            symbols=e["symbols"],
                            symbol_descs=e["symbol_descs"],
                            symbol_properties=e["symbol_properties"],
                            expression=e["expression"],
                        ),
                        samples=samples,
                    )
                )
        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}

        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}

    @property
    def name(self):
        return self._dataset_identifier


class MatSciDataModule(BaseSynthDataModule):
    def __init__(self, root, hf_local_path: Optional[str] = None):
        super().__init__("matsci", "MatSci", root, hf_local_path=hf_local_path)


class ChemReactKineticsDataModule(BaseSynthDataModule):
    def __init__(self, root, hf_local_path: Optional[str] = None):
        super().__init__(
            "chem_react",
            "CRK",
            root,
            default_symbols=["dA_dt", "t", "A"],
            default_symbol_descs=[
                "Rate of change of concentration in chemistry reaction kinetics",
                "Time",
                "Concentration at time t",
            ],
            hf_local_path=hf_local_path,
        )


class BioPopGrowthDataModule(BaseSynthDataModule):
    def __init__(self, root, hf_local_path: Optional[str] = None):
        super().__init__(
            "bio_pop_growth",
            "BPG",
            root,
            default_symbols=["dP_dt", "t", "P"],
            default_symbol_descs=["Population growth rate", "Time", "Population at time t"],
            hf_local_path=hf_local_path,
        )


class PhysOscilDataModule(BaseSynthDataModule):
    def __init__(self, root, hf_local_path: Optional[str] = None):
        super().__init__(
            "phys_osc",
            "PO",
            root,
            default_symbols=["dv_dt", "x", "t", "v"],
            default_symbol_descs=[
                "Acceleration in Nonl-linear Harmonic Oscillator",
                "Position at time t",
                "Time",
                "Velocity at time t",
            ],
            hf_local_path=hf_local_path,
        )


def get_datamodule(name, root_folder, hf_local_path: Optional[str] = None):
    if name == "bio_pop_growth":
        root = root_folder or "datasets/lsr-synth-bio"
        return BioPopGrowthDataModule(root, hf_local_path=hf_local_path)
    elif name == "chem_react":
        root = root_folder or "datasets/lsr-synth-chem"
        return ChemReactKineticsDataModule(root, hf_local_path=hf_local_path)
    elif name == "matsci":
        root = root_folder or "datasets/lsr-synth-matsci"
        return MatSciDataModule(root, hf_local_path=hf_local_path)
    elif name == "phys_osc":
        root = root_folder or "datasets/lsr-synth-phys"
        return PhysOscilDataModule(root, hf_local_path=hf_local_path)
    # elif name == 'feynman':
    #     return FeynmanDataModule()
    elif name == "lsrtransform":
        return TransformedFeynmanDataModule(hf_local_path=hf_local_path)
    else:
        raise ValueError(f"Unknown datamodule name: {name}")
