"""
Base functionality for hfdol.

This module provides Mapping-based interfaces to HuggingFace datasets, models, spaces, and papers,
allowing you to interact with them using familiar Python dictionary operations.

Key Classes:
- HfDatasets: A Mapping interface for browsing and loading HuggingFace datasets
- HfModels: A Mapping interface for browsing and downloading HuggingFace models
- HfSpaces: A Mapping interface for browsing and accessing HuggingFace Spaces
- HfPapers: A Mapping interface for browsing and accessing HuggingFace Papers

All classes provide a unified API for local cached items (via iteration and key access)
and remote searching/downloading capabilities.
"""

import os
from functools import lru_cache
from collections.abc import Mapping
from inspect import Parameter, Signature, signature
from enum import Enum

from datasets import load_dataset
from huggingface_hub import (
    scan_cache_dir,
    snapshot_download,
    HfApi,
    list_models,
    list_datasets,
    list_spaces,
    list_papers,
    space_info,
    paper_info,
)
from huggingface_hub.hf_api import DatasetInfo, ModelInfo, SpaceInfo, PaperInfo


class RepoType(str, Enum):
    """Valid HuggingFace repository types."""

    DATASET = 'dataset'
    MODEL = 'model'
    SPACE = 'space'
    PAPER = 'paper'


DFLT_SIZE_UNIT_BYTES: int = 1024**3  # GiB
HFAPI_CONFIG = {}


def sign_kwargs_with(source_func, first_arg_index=0):
    """
    Minimal decorator to replace **kwargs with parameters from source_func.

    :param source_func: Function whose parameters will be injected
    :param first_arg_index: Index of first parameter to include from source (default 0)

    >>> def source(x=0, y=1, z=2): ...
    >>> @sign_kwargs_with(source)
    ... def target(some, thing=None, **kwargs):
    ...     pass
    >>> str(signature(target))
    '(some, thing=None, *, x=0, y=1, z=2)'

    >>> @sign_kwargs_with(source, first_arg_index=1)
    ... def target2(some, thing=None, **kwargs):
    ...     pass
    >>> str(signature(target2))
    '(some, thing=None, *, y=1, z=2)'
    """

    def change_signature_of_variadic(target_func):
        target_sig = signature(target_func)
        source_sig = signature(source_func)

        # Get target params without **kwargs
        target_params = [
            p for p in target_sig.parameters.values() if p.kind != Parameter.VAR_KEYWORD
        ]
        target_names = {p.name for p in target_params}

        # Get source params starting from first_arg_index
        source_params = list(source_sig.parameters.values())[first_arg_index:]

        # Add source params not in target as keyword-only (preserve VAR_KEYWORD if present)
        new_params = target_params + [
            (
                p
                if p.kind == Parameter.VAR_KEYWORD
                else p.replace(kind=Parameter.KEYWORD_ONLY)
            )
            for p in source_params
            if p.name not in target_names
        ]

        target_func.__signature__ = Signature(new_params)
        return target_func

    return change_signature_of_variadic


# Note: Needs manual work since list_papers doesn't ressemble other list_* functions
def _list_papers(filter=None, *, token=None):
    """
    Wrapper to normalize list_papers interface to use 'filter' parameter like other search functions.
    Note: list_papers only supports 'query' and 'token' parameters, so other kwargs are filtered out.
    """
    return list_papers(query=filter, token=token)


# Single Source of Truth for repo type configurations
repo_type_helpers = dict(
    dataset=dict(
        loader_func=load_dataset,
        search_func=list_datasets,
    ),
    model=dict(
        loader_func=snapshot_download,
        search_func=list_models,
    ),
    space=dict(
        loader_func=space_info,
        search_func=list_spaces,
    ),
    paper=dict(
        loader_func=paper_info,
        search_func=_list_papers,
    ),
)


# Note: lru_cache brings HfApi instantiation from 300ns to 30ns.
#   Not important, but _get_hf_api also centralizes configured instantiation.
@lru_cache(maxsize=1)
def _get_hf_api():
    """"""
    return HfApi(**HFAPI_CONFIG)


def ensure_id(obj):
    if isinstance(obj, (DatasetInfo, ModelInfo, SpaceInfo, PaperInfo)):
        return obj.id
    elif isinstance(obj, str):
        return obj
    else:
        raise ValueError(f"Cannot ensure ID from object of type {type(obj)}")


def get_size(
    repo_id: str, *, unit_bytes: int = DFLT_SIZE_UNIT_BYTES, repo_type: RepoType
) -> float:
    """
    Calculates the total size of a Hugging Face repository (model, dataset, or space) in GiB.

    Args:
        repo_id (str): The ID of the repository (e.g., "bert-base-uncased", "MMMU/MMMU", or "spaces/gradio/chatbot").
        unit_bytes (int): Number of bytes in the desired unit. Default is 1024**3 for GiB. For bytes, enter 1.
        repo_type (RepoType): Type of repository ("model", "dataset", "space"). Required parameter.
                        Papers don't have file sizes, so they're not supported.

    Returns:
        float: The total repository size in the specified unit.

    Raises:
        ValueError: If repo_type is "paper" (papers don't have file sizes) or if repo_type is invalid.
    """
    # Convert enum to string if needed
    repo_type_str = repo_type.value if hasattr(repo_type, 'value') else str(repo_type)

    if repo_type_str == "paper":
        raise ValueError("Papers don't have file sizes - they are metadata objects")

    if repo_type_str not in repo_type_helpers:
        raise ValueError(
            f"Invalid repo_type: {repo_type_str}. Must be one of: {list(repo_type_helpers.keys())}"
        )

    api = _get_hf_api()
    repo_id = ensure_id(repo_id)

    # Use repo_type directly based on validation
    if repo_type_str == "model":
        info = api.model_info(repo_id=repo_id, files_metadata=True)
    elif repo_type_str == "dataset":
        info = api.dataset_info(repo_id=repo_id, files_metadata=True)
    elif repo_type_str == "space":
        info = api.space_info(repo_id=repo_id, files_metadata=True)

    is_not_none = lambda s: s is not None
    total_size_bytes = sum(
        filter(is_not_none, (sibling.size for sibling in info.siblings))
    )
    return total_size_bytes / unit_bytes


def ensure_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    return dirpath


def list_local_repos(repo_type):
    """
    Dynamically list locally cached repositories of a given type using scan_cache_dir.

    Args:
        repo_type (str): The type of repository ("dataset", "model", "space", or "paper").

    Returns:
        list: A list of repo IDs for repositories of the specified type.
    """
    cache_info = scan_cache_dir()
    return [repo.repo_id for repo in cache_info.repos if repo.repo_type == repo_type]


def _create_search_method(search_func):
    """Create a search method with the signature of the given search_func."""

    @sign_kwargs_with(search_func)
    def search(self, filter, **kwargs):
        """
        Search for remote repositories that match the query.

        Args:
            filter: Search query string.
            **kwargs: Additional search parameters. For details, see below.

        Returns:
            Generator of repository info objects.

        For documentation on search params you can use in `**kwargs`, see:
        - datasets: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_datasets
        - models: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_models
        - spaces: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_spaces
        - papers: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_papers
        """
        return self.search_func(filter=filter, **kwargs)

    return search


class HfMapping(Mapping):
    """
    Abstract base class for HuggingFace Mappings.

    This class provides common functionality for datasets, models, spaces, and papers,
    parameterized by repo_type which determines loader_func and search_func.

    Can be used in two ways:
    1. Via subclasses (HfDatasets, HfModels, etc.) for common types - best UX
    2. Via direct parameterization for less common types or dynamic use
    """

    repo_type = None  # To be overridden in subclasses

    def __init__(self, repo_type: RepoType = None):
        # Support both parameterized usage and subclass usage
        if repo_type is not None:
            # Convert enum to string value or use string directly
            self.repo_type = (
                repo_type.value if hasattr(repo_type, 'value') else str(repo_type)
            )
        elif self.repo_type is None:
            raise NotImplementedError(
                "repo_type must be defined in subclass or passed to constructor"
            )
        else:
            # Convert class attribute enum to string if needed
            self.repo_type = (
                self.repo_type.value
                if hasattr(self.repo_type, 'value')
                else str(self.repo_type)
            )

        if self.repo_type not in repo_type_helpers:
            raise ValueError(f"Unsupported repo_type: {self.repo_type}")

        # Get configuration from SSOT
        config = repo_type_helpers[self.repo_type]
        self.loader_func = staticmethod(config["loader_func"])
        self.search_func = staticmethod(config["search_func"])

        # Create search method with correct signature and bind it to this instance
        search_method = _create_search_method(config["search_func"])
        self.search = search_method.__get__(self, type(self))

    def __getitem__(self, key):
        """Load/download an item using the configured loader function."""
        return self.loader_func(ensure_id(key))

    def _keys(self):
        """Get list of local repository IDs of the configured type."""
        return list_local_repos(self.repo_type)

    def __iter__(self):
        """Iterate through keys to local repositories."""
        return iter(self._keys())

    def __len__(self):
        """Number of local repositories."""
        return len(self._keys())

    def get_size(self, key: str, *, unit_bytes: int = DFLT_SIZE_UNIT_BYTES) -> float:
        """Get size (by default, in GiB) of an item from it's key (repo ID)"""
        return get_size(key, unit_bytes=unit_bytes, repo_type=self.repo_type)

    # Note: search method is dynamically created in __init__ with the correct signature


class HfDatasets(HfMapping):
    """
    A Mapping interface to HuggingFace datasets.

    Provides dictionary-like access to locally cached datasets and seamless
    downloading of remote datasets. Keys are dataset repository IDs (e.g., 'stingning/ultrachat').
    Values are loaded dataset objects from the datasets library.

    Examples:

    >>> d = HfDatasets()

    List locally cached datasets:

    >>> list_of_dataset_repo_ids = list(d)

    Search remote datasets

    >>> results = d.search('music', gated=False)  # doctest: +SKIP

    Load or download a dataset

    >>> data = d['some/dataset']  # doctest: +SKIP

    """

    repo_type = RepoType.DATASET


class HfModels(HfMapping):
    """
    A Mapping interface to HuggingFace models.

    Provides dictionary-like access to locally cached models and seamless
    downloading of remote models. Keys are model repository IDs (e.g., 'sentence-transformers/all-MiniLM-L6-v2').
    Values are file paths to the downloaded model directories.

    Examples:

    >>> m = HfModels()

    List locally cached models

    >>> list_of_model_repo_ids = list(m)

    Search remote models

    >>> results = m.search('embeddings', gated=False)  # doctest: +SKIP

    Download or get path to a model

    >>> model_path = m['some/model']  # doctest: +SKIP

    """

    repo_type = RepoType.MODEL


class HfSpaces(HfMapping):
    """
    A Mapping interface to HuggingFace Spaces.

    Provides dictionary-like access to locally cached spaces and seamless
    retrieval of remote space information. Keys are space repository IDs (e.g., 'gradio/chatbot').
    Values are SpaceInfo objects containing space metadata and configuration.

    Examples:

    >>> s = HfSpaces()

    List locally cached spaces

    >>> list_of_space_repo_ids = list(s)

    Search remote spaces

    >>> results = s.search('gradio', gated=False)  # doctest: +SKIP

    Get information about a space

    >>> space_info = s['gradio/chatbot']  # doctest: +SKIP

    """

    repo_type = RepoType.SPACE


class HfPapers(HfMapping):
    """
    A Mapping interface to HuggingFace Papers.

    Provides dictionary-like access to paper information. Keys are paper IDs.
    Values are PaperInfo objects containing paper metadata, abstracts, and links.

    Note: Papers are metadata objects only - they don't have downloadable files or sizes.

    Examples:

    >>> p = HfPapers()

    Search papers

    >>> results = p.search('transformer')  # doctest: +SKIP

    Get information about a paper

    >>> paper_info = p['2017.12345']  # doctest: +SKIP

    """

    repo_type = RepoType.PAPER


# ------------------------------------------------------------------------------------
# Convenience instances (lowercase by convention)
datasets = HfDatasets()
models = HfModels()
spaces = HfSpaces()
papers = HfPapers()
