from collections.abc import Iterator, Sequence
import logging
import math
import multiprocessing
import os
import random
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
from openpi.shared import array_typing as at
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.dataset_zarr import MultiZarrDataset


T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def _unwrap_dataset(dataset: Dataset[T_co]) -> Dataset[T_co]:
    """Return the innermost dataset by stripping transformation wrappers."""
    visited_ids: set[int] = set()
    while True:
        inner = getattr(dataset, "_dataset", None)
        if inner is None or inner is dataset or id(inner) in visited_ids:
            break
        visited_ids.add(id(dataset))
        dataset = inner
    return dataset


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, 
    action_horizon: int, 
    model_config: _model.BaseModelConfig,
    sample_ratio: float = 1.0,
) -> tuple[Dataset, Dataset | None]:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    val_dataset = None
    if hasattr(data_config, 'dataset_config_path') and data_config.dataset_config_path:
        # MultiZarrDataset
        train_dataset = MultiZarrDataset(data_config, model_config.action_horizon, split='train') 
        if sample_ratio < 1.0:
            print(f"Training Dataset sample ratio: {sample_ratio}")
            train_dataset.set_sample_ratio(sample_ratio)
        if data_config.use_val_dataset:
            val_dataset = train_dataset.get_val_dataset()
            if val_dataset is not None and sample_ratio < 1.0:
                print(f"Validation dataset sample ratio: {sample_ratio}")
                val_dataset.set_sample_ratio(sample_ratio)
    else:
        # Pi LeRobotDataset
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        train_dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
        )

        if data_config.prompt_from_task:
            train_dataset = TransformedDataset(train_dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return train_dataset, val_dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    from openpi.training.droid_rlds_dataset import DroidRldsDataset
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, 
                      data_config: _config.DataConfig, *, 
                      skip_norm_stats: bool = False,
                      only_model_transforms: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    """
    For only_model_transforms is True, 
        the data transforms & normalization will be skipped by the PI-transform-chain, but be conducted inside the original dataset.
    """
    if only_model_transforms:
        return TransformedDataset(
            dataset, [*data_config.model_transforms.inputs,],
        )

    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=(data_config.norm_type == 'quantile')),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=(data_config.norm_type == 'quantile')),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> tuple[DataLoader[tuple[_model.Observation, _model.Actions]], 
          DataLoader[tuple[_model.Observation, _model.Actions]] | None]:  # return (train_data_loader, val_data_loader)
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if hasattr(data_config, 'rlds_data_dir') and data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )  # return train_data_loader, val_data_loader


def _create_torch_data_loader(train_dataset: Dataset,
                              data_config: _config.DataConfig,
                              batch_size: int,
                              shuffle: bool,
                              num_batches: int | None,
                              num_workers: int,
                              seed: int,
                              framework: str,
                              domain_batch_split: bool,
                              sharding: jax.sharding.Sharding | None = None,):
    # Use TorchDataLoader for both frameworks
    sampler = None
    batch_sampler = None
    world_size = 1
    rank = 0
    effective_domain_batch_split = domain_batch_split
    base_dataset = None

    if effective_domain_batch_split:
        base_dataset = _unwrap_dataset(train_dataset)
        if not isinstance(base_dataset, MultiZarrDataset):
            raise ValueError("domain_batch_split requires a MultiZarrDataset instance.")

        domain_cumsum = np.asarray(base_dataset.domain_dataset_length_cumsum)
        # domain_dataset_length_cumsum = [0, len(domain0), len(domain0)+len(domain1), ...]
        # so diff gives per-domain sample counts.
        domain_lengths = np.diff(domain_cumsum)
        active_domain_count = int(np.count_nonzero(domain_lengths > 0))
        if active_domain_count <= 1:
            effective_domain_batch_split = False
            logging.info(
                "domain_batch_split requested but detected %d active domain(s); "
                "auto-disabling domain_batch_split for this loader.",
                active_domain_count,
            )

    if framework == "pytorch":
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            if not effective_domain_batch_split:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    drop_last=True,
                )
        local_batch_size = batch_size // world_size
    else:
        world_size = jax.process_count()
        rank = jax.process_index()
        local_batch_size = batch_size // world_size

    if local_batch_size <= 0:
        raise ValueError("Local batch size must be positive; increase batch_size or reduce world_size.")

    if effective_domain_batch_split:
        if batch_size % world_size != 0:
            raise ValueError(f"domain_batch_split requires batch_size {batch_size} to be divisible by world_size {world_size}.")
        assert isinstance(base_dataset, MultiZarrDataset)
        batch_sampler = DomainBatchSampler(
            dataset=base_dataset,
            batch_size=local_batch_size,
            shuffle=shuffle,
            seed=seed,
            world_size=world_size,
            rank=rank,
        )

    logging.info(f"local_batch_size: {local_batch_size}")
    train_data_loader = TorchDataLoader(
        train_dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    train_data_loader = DataLoaderImpl(data_config, train_data_loader)
    return train_data_loader


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    val_num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    load_zarr_norm_stats: bool = False,
    domain_batch_split: bool = False,
) -> tuple[DataLoader[tuple[_model.Observation, _model.Actions]], 
           DataLoader[tuple[_model.Observation, _model.Actions]] | None]:  # return (train_data_loader, val_data_loader)
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    train_dataset, val_dataset = create_torch_dataset(data_config, action_horizon, model_config)
    if load_zarr_norm_stats:
        train_dataset.load_norm_stats(independent_norm_mode=data_config.independent_norm_mode, 
                                      norm_time_dim=data_config.norm_time_dim, 
                                      norm_type=data_config.norm_type)
        if val_dataset is not None:
            val_dataset.load_norm_stats(independent_norm_mode=data_config.independent_norm_mode, 
                                        norm_time_dim=data_config.norm_time_dim, 
                                        norm_type=data_config.norm_type)
    
    only_model_transforms = hasattr(data_config, 'dataset_config_path')
    
    train_dataset = transform_dataset(train_dataset, data_config, 
                                      skip_norm_stats=skip_norm_stats,
                                      only_model_transforms=only_model_transforms)
    train_data_loader = _create_torch_data_loader(train_dataset, data_config, batch_size, shuffle, num_batches, num_workers, seed, framework, 
                                                  domain_batch_split=domain_batch_split, sharding=sharding)
    if val_dataset is not None:
        val_dataset = transform_dataset(val_dataset, data_config, 
                                        skip_norm_stats=skip_norm_stats,
                                        only_model_transforms=only_model_transforms)
        # Check if validation dataset is empty before creating data loader
        val_dataset_len = len(val_dataset)
        if val_dataset_len == 0:
            logging.warning("Validation dataset is empty, skipping validation data loader creation.")
            val_data_loader = None
        else:
            val_data_loader = _create_torch_data_loader(val_dataset, data_config, batch_size, shuffle=False, num_batches=num_batches, num_workers=val_num_workers, seed=seed, framework=framework, 
                                                        domain_batch_split=domain_batch_split, sharding=sharding)
    else:
        val_data_loader = None
    return train_data_loader, val_data_loader


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> tuple[DataLoader[tuple[_model.Observation, _model.Actions]], 
          DataLoader[tuple[_model.Observation, _model.Actions]] | None]:  # return (train_data_loader, val_data_loader)
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader), None


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler[list[int]] | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {
            "dataset": typing.cast(torch.utils.data.Dataset, dataset),
            "num_workers": num_workers,
            "multiprocessing_context": mp_context,
            "persistent_workers": num_workers > 0,
            "pin_memory": torch.cuda.is_available(),  # Speed up CPU->GPU transfer
            # prefetch_factor defaults to 2 when num_workers > 0, None when num_workers = 0, will be set by default.
            "collate_fn": _collate_fn,
            "worker_init_fn": _worker_init_fn,
            "generator": generator,
        }
        if batch_sampler is not None:
            # When using batch_sampler, batch_size, shuffle, sampler, and drop_last are mutually exclusive
            loader_kwargs["batch_sampler"] = batch_sampler
        else:
            loader_kwargs["batch_size"] = local_batch_size
            loader_kwargs["shuffle"] = (sampler is None and shuffle)
            loader_kwargs["sampler"] = sampler
            loader_kwargs["drop_last"] = True

        self._data_loader = torch.utils.data.DataLoader(**loader_kwargs)
        self._batch_sampler = batch_sampler
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the batch sampler (if it supports it)."""
        self._epoch = epoch
        if self._batch_sampler is not None and hasattr(self._batch_sampler, 'set_epoch'):
            self._batch_sampler.set_epoch(epoch)

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            # Set epoch before creating new iterator to ensure all workers use the same epoch
            if self._batch_sampler is not None and hasattr(self._batch_sampler, 'set_epoch'):
                self._batch_sampler.set_epoch(self._epoch)
            
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, convert to torch tensors (skip strings)
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    # Convert to torch tensors, but keep strings as lists/arrays
                    def _to_tensor_or_keep(x):
                        # Keep strings and other non-numeric types as-is
                        if isinstance(x, (str, bytes)):
                            return x
                        # Check if it's a numpy array with string dtype
                        if isinstance(x, np.ndarray) and (x.dtype.kind == 'U' or x.dtype.kind == 'S'):
                            return x  # Keep string arrays as numpy arrays
                        # Try to convert to tensor
                        try:
                            return torch.as_tensor(x)
                        except (ValueError, TypeError):
                            # If conversion fails, return as-is (e.g., for lists of strings)
                            return x
                    
                    yield jax.tree.map(_to_tensor_or_keep, batch)
            
            # Increment epoch for next iteration
            self._epoch += 1


class DomainBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield batches of indices that belong to a single domain.
    
    Logic:
    1. For each domain, generate all possible batches (grouped by batch_size)
    2. Merge all batches from all domains (automatically preserves domain size proportion)
    3. Shuffle all batches each epoch for randomness
    4. Distribute batches evenly across ranks
    """

    def __init__(
        self,
        dataset: MultiZarrDataset,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if world_size <= 0:
            raise ValueError("world_size must be positive.")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size).")
        
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._world_size = world_size
        self._rank = rank
        
        # Get domain ranges and sizes
        domain_cumsum = list(map(int, dataset.domain_dataset_length_cumsum.tolist()))
        self._domain_ranges = []
        self._domain_sizes = []
        for start, end in zip(domain_cumsum, domain_cumsum[1:]):
            if end > start:
                self._domain_ranges.append((start, end))
                self._domain_sizes.append(end - start)
        if not self._domain_ranges:
            dataset_len = len(dataset)
            domain_names = getattr(dataset, 'domain_list', [])
            raise ValueError(
                f"No domains with data were found for domain_batch_split. "
                f"Total dataset length: {dataset_len}, "
                f"Domain names: {domain_names}, "
                f"Domain lengths: {[end - start for start, end in zip(domain_cumsum, domain_cumsum[1:])]}. "
                f"This usually happens when the validation dataset is empty (e.g., val_ratio is too small for the dataset size)."
            )
        
        # Calculate total number of batches (sum of batches from all domains)
        # Each domain generates ceil(domain_size / batch_size) batches
        self._total_batches = sum(math.ceil(size / batch_size) for size in self._domain_sizes)
        
        # Store domain information for batch generation
        self._domain_indices = [list(range(start, end)) for start, end in self._domain_ranges]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler."""
        self._epoch = epoch

    def __len__(self) -> int:
        """Number of batches for this rank."""
        base = self._total_batches // self._world_size
        extra = self._total_batches % self._world_size
        return base + (1 if self._rank < extra else 0)

    def _generate_all_batches(self) -> list[list[int]]:
        """Generate all batches: group by domain, then merge all batches."""
        all_batches = []
        
        # For each domain, generate all possible batches
        for domain_idx, domain_indices in enumerate(self._domain_indices):
            # Shuffle domain indices for this epoch
            if self._shuffle:
                domain_order = domain_indices[:]
                rng_domain = random.Random(self._seed + self._epoch + domain_idx)
                rng_domain.shuffle(domain_order)
            else:
                domain_order = domain_indices[:]
            
            # Generate all full batches from this domain
            for i in range(0, len(domain_order), self._batch_size):
                batch = domain_order[i:i + self._batch_size]
                if len(batch) == self._batch_size:
                    all_batches.append(batch)
            
            # Handle incomplete last batch (edge case: only if domain data is exhausted)
            remaining = len(domain_order) % self._batch_size
            if remaining > 0:
                last_batch_start = (len(domain_order) // self._batch_size) * self._batch_size
                incomplete_batch = domain_order[last_batch_start:]
                # Edge case: fill incomplete batch by wrapping around from the beginning of incomplete_batch
                # This ensures we only repeat samples that are already in the incomplete batch,
                # maintaining data consistency (e.g., timestamp=0 positions remain correct)
                # Store the original remaining samples to cycle through
                original_remaining = incomplete_batch[:]
                while len(incomplete_batch) < self._batch_size:
                    # Wrap around: cycle through the original remaining samples
                    # This preserves the relative order and ensures we're repeating nearby samples
                    # rather than randomly selecting from the entire domain
                    idx_in_cycle = (len(incomplete_batch) - remaining) % len(original_remaining)
                    incomplete_batch.append(original_remaining[idx_in_cycle])
                all_batches.append(incomplete_batch)
        
        return all_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Generate batches for this rank."""
        # Generate all batches
        all_batches = self._generate_all_batches()
        
        # Shuffle all batches for randomness (if enabled)
        if self._shuffle:
            rng = random.Random(self._seed + self._epoch)
            rng.shuffle(all_batches)
        
        # Distribute batches evenly across ranks
        for batch_idx, batch in enumerate(all_batches):
            if batch_idx % self._world_size == self._rank:
                yield batch


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    # Handle string types separately - they should remain as lists, not be stacked
    def _stack_or_list(*xs):
        # Check if all elements are strings or if we can't convert to array
        try:
            # Try to convert to numpy array and stack
            arrays = [np.asarray(x) for x in xs]
            # Check if any is string type
            if any(arr.dtype.kind == 'U' or arr.dtype.kind == 'S' for arr in arrays if hasattr(arr, 'dtype')):
                # Return as list for string types
                return list(xs)
            return np.stack(arrays, axis=0)
        except (ValueError, TypeError):
            # If stacking fails (e.g., for strings), return as list
            return list(xs)
    
    return jax.tree.map(_stack_or_list, *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __len__(self) -> int:
        """Return the number of batches for this data loader."""
        # For TorchDataLoader, try to get length from batch_sampler or torch_loader
        if isinstance(self._data_loader, TorchDataLoader):
            if hasattr(self._data_loader, '_batch_sampler') and self._data_loader._batch_sampler is not None:
                # Use batch_sampler length (for domain_batch_split)
                return len(self._data_loader._batch_sampler)
            elif hasattr(self._data_loader, 'torch_loader'):
                # Use torch DataLoader length
                return len(self._data_loader.torch_loader)
            else:
                raise NotImplementedError("Cannot determine length of TorchDataLoader without batch_sampler or torch_loader")
        elif isinstance(self._data_loader, RLDSDataLoader):
            # For RLDSDataLoader, use num_batches if available
            if self._data_loader._num_batches is not None:
                return self._data_loader._num_batches
            else:
                raise NotImplementedError("Cannot determine length of RLDSDataLoader without num_batches")
        else:
            raise NotImplementedError(f"Cannot determine length for data loader type: {type(self._data_loader)}")

    def __iter__(self):
        for batch in self._data_loader:
            if 'actions' in batch.keys():
                with at.disable_typechecking():
                    yield _model.Observation.from_dict(batch), batch["actions"]
            elif 'action' in batch.keys():
                # Backward compatibility: support old 'action' key
                with at.disable_typechecking():
                    yield _model.Observation.from_dict(batch), batch["action"]
            else:
                raise ValueError(f"'actions' or 'action' not found in batch keys: {batch.keys()}")
