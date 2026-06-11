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


def _load_synth_split_from_local_parquet(dataset_dir: Path, dataset_identifier: str):
    """Load metadata rows from a local Hub snapshot (data/lsr_synth_<id>-*.parquet)."""
    parquet_files = sorted(dataset_dir.glob(f"data/lsr_synth_{dataset_identifier}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Missing parquet under local snapshot: {dataset_dir}/data/lsr_synth_{dataset_identifier}-*.parquet"
        )
    ds_dict = datasets.load_dataset("parquet", data_files=[str(p) for p in parquet_files])
    return ds_dict["train"]

class TransformedFeynmanDataModule:
    def __init__(self, local_snapshot_dir: Optional[str] = None):
        self._dataset_dir = None
        self._dataset_identifier = 'lsr_transform'
        self._local_snapshot_dir = local_snapshot_dir

    def setup(self):
        if self._local_snapshot_dir is not None:
            self._dataset_dir = Path(self._local_snapshot_dir).expanduser().resolve()
            h5 = self._dataset_dir / "lsr_bench_data.hdf5"
            if not h5.is_file():
                raise FileNotFoundError(f"Missing file in local snapshot: {h5}")
            parquet_files = sorted(self._dataset_dir.glob("data/lsr_transform-*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"Missing parquet in local snapshot: {self._dataset_dir}/data/lsr_transform-*.parquet"
                )
            ds = datasets.load_dataset("parquet", data_files=[str(p) for p in parquet_files])[
                "train"
            ]
        else:
            self._dataset_dir = Path(_download(repo_id=REPO_ID))
            ds = datasets.load_dataset(REPO_ID)['lsr_transform']
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        self.problems = []
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {k:v[...].astype(np.float64) for k,v in sample_file[f'/lsr_transform/{e["name"]}'].items()}
                self.problems.append(Problem(dataset_identifier=self._dataset_identifier,
                                        equation_idx = e['name'],
                                        gt_equation=Equation(
                                            symbols=e['symbols'],
                                            symbol_descs=e['symbol_descs'],
                                            symbol_properties=e['symbol_properties'],
                                            expression=e['expression'],
                                        ),
                                        samples=samples)
                )
        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}

    @property
    def name(self):
        return "LSR_Transform"

class SynProblem(Problem):
    @property
    def train_samples(self):
        return self.samples['train_data']
    
    @property
    def test_samples(self):
        return self.samples['id_test_data']
    
    @property
    def ood_test_samples(self):
        return self.samples['ood_test_data']

class BaseSynthDataModule:
    def __init__(
        self,
        dataset_identifier,
        short_dataset_identifier,
        root,
        default_symbols=None,
        default_symbol_descs=None,
        local_snapshot_dir: Optional[str] = None,
    ):
        self._dataset_dir = Path(root)
        self._dataset_identifier = dataset_identifier
        self._short_dataset_identifier = short_dataset_identifier
        self._default_symbols = default_symbols
        self._default_symbol_descs = default_symbol_descs
        self._local_snapshot_dir = local_snapshot_dir

    def setup(self):
        if self._local_snapshot_dir is not None:
            self._dataset_dir = Path(self._local_snapshot_dir).expanduser().resolve()
            h5 = self._dataset_dir / "lsr_bench_data.hdf5"
            if not h5.is_file():
                raise FileNotFoundError(f"Missing file in local snapshot: {h5}")
            ds = _load_synth_split_from_local_parquet(self._dataset_dir, self._dataset_identifier)
        else:
            self._dataset_dir = Path(_download(repo_id=REPO_ID))
            ds = datasets.load_dataset(REPO_ID)[f'lsr_synth_{self._dataset_identifier}']
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        self.problems = []
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {k:v[...].astype(np.float64) for k,v in sample_file[f'/lsr_synth/{self._dataset_identifier}/{e["name"]}'].items()}
                self.problems.append(Problem(dataset_identifier=self._dataset_identifier,
                                        equation_idx = e['name'],
                                        gt_equation=Equation(
                                            symbols=e['symbols'],
                                            symbol_descs=e['symbol_descs'],
                                            symbol_properties=e['symbol_properties'],
                                            expression=e['expression'],
                                        ),
                                        samples=samples)
                )
        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}

    
        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}

    @property
    def name(self):
        return self._dataset_identifier

class MatSciDataModule(BaseSynthDataModule):
    def __init__(self, root, local_snapshot_dir: Optional[str] = None):
        super().__init__("matsci", "MatSci", root, local_snapshot_dir=local_snapshot_dir)

class ChemReactKineticsDataModule(BaseSynthDataModule):
    def __init__(self, root, local_snapshot_dir: Optional[str] = None):
        super().__init__("chem_react", "CRK", root,
                         default_symbols=['dA_dt', 't', 'A'],
                         default_symbol_descs=['Rate of change of concentration in chemistry reaction kinetics', 'Time', 'Concentration at time t'],
                         local_snapshot_dir=local_snapshot_dir)
        
class BioPopGrowthDataModule(BaseSynthDataModule):
    def __init__(self, root, local_snapshot_dir: Optional[str] = None):
        super().__init__("bio_pop_growth", "BPG", root,
                         default_symbols=['dP_dt', 't', 'P'],
                         default_symbol_descs=['Population growth rate', 'Time', 'Population at time t'],
                         local_snapshot_dir=local_snapshot_dir)
        
class PhysOscilDataModule(BaseSynthDataModule):
    def __init__(self, root, local_snapshot_dir: Optional[str] = None):
        super().__init__("phys_osc", "PO", root,
                         default_symbols=['dv_dt', 'x', 't', 'v'],
                         default_symbol_descs=['Acceleration in Nonl-linear Harmonic Oscillator', 'Position at time t', 'Time', 'Velocity at time t'],
                         local_snapshot_dir=local_snapshot_dir)

def get_datamodule(name, root_folder, local_snapshot_dir: Optional[str] = None):
    if name == 'bio_pop_growth':
        root = root_folder or "datasets/lsr-synth-bio"
        return BioPopGrowthDataModule(root, local_snapshot_dir=local_snapshot_dir)
    elif name == 'chem_react':
        root = root_folder or "datasets/lsr-synth-chem"
        return ChemReactKineticsDataModule(root, local_snapshot_dir=local_snapshot_dir)
    elif name == 'matsci':
        root = root_folder or "datasets/lsr-synth-matsci"
        return MatSciDataModule(root, local_snapshot_dir=local_snapshot_dir)
    elif name == 'phys_osc':
        root = root_folder or "datasets/lsr-synth-phys"
        return PhysOscilDataModule(root, local_snapshot_dir=local_snapshot_dir)
    # elif name == 'feynman':
    #     return FeynmanDataModule()
    elif name == 'lsrtransform':
        return TransformedFeynmanDataModule(local_snapshot_dir=local_snapshot_dir)
    else:
        raise ValueError(f"Unknown datamodule name: {name}")