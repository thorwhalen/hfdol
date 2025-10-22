"""Tests for hfdol base.py"""

import pytest
import typing
from hfdol.base import (
    datasets,
    models,
    spaces,
    papers,
    HfDatasets,
    HfModels,
    HfSpaces,
    HfPapers,
    RepoType,
    list_local_repos,
    ensure_id,
    get_size,
    repo_type_helpers,
)
from huggingface_hub.hf_api import DatasetInfo, ModelInfo, SpaceInfo, PaperInfo


def test_ensure_id():
    """Test the ensure_id function with different input types."""
    # Test with string
    assert ensure_id("some/repo") == "some/repo"

    # Test with DatasetInfo - create a real one or mock properly
    # For now, just test the error case

    # Test with invalid type
    with pytest.raises(ValueError):
        ensure_id(123)


def test_repo_type_helpers_ssot():
    """Test that the repo_type_helpers SSOT configuration is correct."""
    expected_types = {"dataset", "model", "space", "paper"}
    assert set(repo_type_helpers.keys()) == expected_types

    # Check each config has required keys
    for repo_type, config in repo_type_helpers.items():
        assert "loader_func" in config
        assert "search_func" in config
        assert callable(config["loader_func"])
        assert callable(config["search_func"])


def test_hf_datasets_class_attributes():
    """Test that HfDatasets has the correct class attributes."""
    from hfdol.base import RepoType

    assert HfDatasets.repo_type == RepoType.DATASET

    # Test that instance has the configured functions
    d = HfDatasets()
    assert hasattr(d, 'loader_func')
    assert hasattr(d, 'search_func')
    assert d.loader_func is not None
    assert d.search_func is not None


def test_hf_models_class_attributes():
    """Test that HfModels has the correct class attributes."""
    from hfdol.base import RepoType

    assert HfModels.repo_type == RepoType.MODEL

    # Test that instance has the configured functions
    m = HfModels()
    assert hasattr(m, 'loader_func')
    assert hasattr(m, 'search_func')
    assert m.loader_func is not None
    assert m.search_func is not None


def test_hf_spaces_class_attributes():
    """Test that HfSpaces has the correct class attributes."""
    from hfdol.base import RepoType

    assert HfSpaces.repo_type == RepoType.SPACE

    # Test that instance has the configured functions
    s = HfSpaces()
    assert hasattr(s, 'loader_func')
    assert hasattr(s, 'search_func')
    assert s.loader_func is not None
    assert s.search_func is not None


def test_hf_papers_class_attributes():
    """Test that HfPapers has the correct class attributes."""
    from hfdol.base import RepoType

    assert HfPapers.repo_type == RepoType.PAPER

    # Test that instance has the configured functions
    p = HfPapers()
    assert hasattr(p, 'loader_func')
    assert hasattr(p, 'search_func')
    assert p.loader_func is not None
    assert p.search_func is not None


def test_hf_datasets_instance():
    """Test basic HfDatasets functionality using the singleton."""
    # Test that the singleton is already instantiated and ready to use
    assert hasattr(datasets, 'repo_type')
    assert datasets.repo_type == "dataset"

    # Test that it's iterable (even if empty)
    list(datasets)  # Should not raise an error

    # Test length
    len(datasets)  # Should not raise an error

    # Test _keys method
    keys = datasets._keys()
    assert isinstance(keys, list)


def test_hf_models_instance():
    """Test basic HfModels functionality using the singleton."""
    # Test that the singleton is already instantiated and ready to use
    assert hasattr(models, 'repo_type')
    assert models.repo_type == "model"

    # Test that it's iterable (even if empty)
    list(models)  # Should not raise an error

    # Test length
    len(models)  # Should not raise an error

    # Test _keys method
    keys = models._keys()
    assert isinstance(keys, list)


def test_hf_spaces_instance():
    """Test basic HfSpaces functionality using the singleton."""
    # Test that the singleton is already instantiated and ready to use
    assert hasattr(spaces, 'repo_type')
    assert spaces.repo_type == "space"

    # Test that it's iterable (even if empty)
    list(spaces)  # Should not raise an error

    # Test length
    len(spaces)  # Should not raise an error

    # Test _keys method
    keys = spaces._keys()
    assert isinstance(keys, list)


def test_hf_papers_instance():
    """Test basic HfPapers functionality using the singleton."""
    # Test that the singleton is already instantiated and ready to use
    assert hasattr(papers, 'repo_type')
    assert papers.repo_type == "paper"

    # Test that it's iterable (even if empty)
    list(papers)  # Should not raise an error

    # Test length
    len(papers)  # Should not raise an error

    # Test _keys method
    keys = papers._keys()
    assert isinstance(keys, list)


def test_list_local_repos():
    """Test the list_local_repos function."""
    # Should return lists for all repo types
    datasets = list_local_repos("dataset")
    models = list_local_repos("model")
    spaces = list_local_repos("space")
    papers = list_local_repos("paper")

    assert isinstance(datasets, list)
    assert isinstance(models, list)
    assert isinstance(spaces, list)
    assert isinstance(papers, list)


def test_get_size_function_signature():
    """Test that get_size has the correct signature and required parameters."""
    import inspect

    sig = inspect.signature(get_size)

    # Should have repo_id as positional, unit_bytes and repo_type as keyword-only
    params = list(sig.parameters.keys())
    assert 'repo_id' in params
    assert 'unit_bytes' in params
    assert 'repo_type' in params

    # repo_type should be required (no default)
    assert sig.parameters['repo_type'].default == inspect.Parameter.empty

    # unit_bytes should have a default value
    assert sig.parameters['unit_bytes'].default is not None


def test_get_size_validation():
    """Test that get_size validates repo_type parameter."""
    # Test with valid repo_types
    from hfdol.base import RepoType

    # Should not raise for valid types (might raise network errors, but not validation errors)
    try:
        get_size("lysandre/test-model", repo_type=RepoType.MODEL)
    except Exception as e:
        # Only network errors are expected, not validation errors
        assert (
            "Repository" in str(e) or "model" in str(e).lower() or "lysandre" in str(e)
        )

    # Test with invalid repo_type
    with pytest.raises(ValueError, match="Invalid repo_type"):
        get_size("some/repo", repo_type="invalid_type")

    # Test with paper repo_type
    with pytest.raises(ValueError, match="Papers don't have file sizes"):
        get_size("some/paper", repo_type=RepoType.PAPER)


def test_hf_mapping_get_size_methods():
    """Test that all singleton instances have get_size methods."""
    # Should have get_size methods
    for instance in [datasets, models, spaces, papers]:
        assert hasattr(instance, 'get_size')
        assert callable(instance.get_size)


def test_datasets_integration():
    """
    Integration test for datasets singleton covering searching, downloading,
    sizing, and local dataset management.
    """
    key1 = "llamafactory/tiny-supervised-dataset"
    key2 = "ucirvine/sms_spam"

    # Test the get_size function (does NOT download the data) - now requires repo_type
    assert round(get_size(key1, repo_type="dataset"), 4) == 0.0001
    # Get size in bytes
    assert get_size(key2, unit_bytes=1, repo_type="dataset") == 365026.0

    # Test search functionality
    search_results = list(datasets.search('tiny', limit=10))
    assert len(search_results) > 0
    assert any('tiny' in result.id.lower() for result in search_results)

    # Test download and load
    val1 = datasets[key1]

    # Test __contains__ - now we should have the key1 in local cache
    assert key1 in datasets

    # Test the contents of val1 are as expected
    assert list(val1) == ['train']
    assert list(val1['train'].features) == ['instruction', 'input', 'output']
    assert val1['train'].num_rows == 300

    # Test that the dataset is now in local listings
    local_datasets = list(datasets)
    assert key1 in local_datasets

    # Test instance get_size method (should use dataset repo_type automatically)
    size_via_instance = datasets.get_size(key1)
    assert round(size_via_instance, 4) == 0.0001


def test_models_integration():
    """
    Integration test for models singleton covering searching, downloading,
    and local model management.
    """
    model_key = "lysandre/test-model"

    # Test the get_size function for a model - now requires repo_type
    model_size = get_size(model_key, repo_type="model")
    assert isinstance(model_size, float)
    assert model_size > 0

    # Test search functionality
    search_results = list(models.search('test', limit=10))
    assert len(search_results) > 0
    assert any('test' in result.id.lower() for result in search_results)

    # Test download - this returns the path to the downloaded model
    model_path = models[model_key]
    assert isinstance(model_path, str)
    assert model_path  # Should be non-empty

    # Test __contains__ - now we should have the model in local cache
    assert model_key in models

    # Test that the model is now in local listings
    local_models = list(models)
    assert model_key in local_models

    # Test instance get_size method (should use model repo_type automatically)
    size_via_instance = models.get_size(model_key)
    assert isinstance(size_via_instance, float)
    assert size_via_instance > 0


def test_cross_type_get_size():
    """Test get_size with explicit repo_type specification."""
    # Test with a known dataset - must specify repo_type
    dataset_size = get_size("ucirvine/sms_spam", repo_type="dataset")
    assert dataset_size == 365026.0 / (1024**3)  # Default unit is GiB

    # Test with a known model - must specify repo_type
    model_size = get_size("lysandre/test-model", repo_type="model")
    assert isinstance(model_size, float)
    assert model_size > 0

    # Test using enum values
    from hfdol.base import RepoType

    explicit_dataset_size = get_size("ucirvine/sms_spam", repo_type=RepoType.DATASET)
    assert explicit_dataset_size == dataset_size

    explicit_model_size = get_size("lysandre/test-model", repo_type=RepoType.MODEL)
    assert explicit_model_size == model_size


def test_get_size_paper_error():
    """Test that get_size raises appropriate error for papers."""
    with pytest.raises(ValueError, match="Papers don't have file sizes"):
        get_size("some_paper_id", repo_type="paper")


def test_spaces_and_papers_integration():
    """
    Integration test for spaces and papers singletons covering searching and info retrieval.
    """
    # Test space search functionality - just check we get results
    space_search_results = list(spaces.search('demo', limit=5))
    assert len(space_search_results) > 0

    # Test paper search functionality
    paper_search_results = list(
        papers.search('transformer')
    )  # Note: No limit arg for papers search!!
    assert len(paper_search_results) > 0
    # Check that at least one result has 'transformer' in title or abstract
    has_transformer = any(
        'transformer' in getattr(result, 'title', '').lower()
        or 'transformer' in getattr(result, 'summary', '').lower()
        for result in paper_search_results
    )
    assert has_transformer

    # Test accessing specific items (if they exist)
    if space_search_results:
        first_space = space_search_results[0]
        space_info = spaces[first_space.id]
        assert space_info is not None
        assert hasattr(space_info, 'id')

    if paper_search_results:
        first_paper = paper_search_results[0]
        paper_info = papers[first_paper.id]
        assert paper_info is not None
        assert hasattr(paper_info, 'id')


def test_repo_type_enum():
    """Test that RepoType enum matches repo_type_helpers keys and supports both enum and string access."""
    from hfdol.base import RepoType

    # Check that RepoType enum values match repo_type_helpers keys
    enum_values = [rt.value for rt in RepoType]
    assert set(enum_values) == set(repo_type_helpers.keys())

    # Check specific expected values
    expected_types = {"dataset", "model", "space", "paper"}
    assert set(enum_values) == expected_types

    # Test that enum supports string comparison (str, Enum inheritance)
    assert RepoType.DATASET == "dataset"
    assert RepoType.MODEL == "model"
    assert RepoType.SPACE == "space"
    assert RepoType.PAPER == "paper"

    # Test that we can get string values from enum
    assert RepoType.DATASET.value == "dataset"
    assert RepoType.MODEL.value == "model"
    assert RepoType.SPACE.value == "space"
    assert RepoType.PAPER.value == "paper"


def test_singleton_instances():
    """Test that the singleton instances are properly configured and ready to use."""
    # Test that all singletons are available and properly configured
    assert hasattr(datasets, 'repo_type') and datasets.repo_type == "dataset"
    assert hasattr(models, 'repo_type') and models.repo_type == "model"
    assert hasattr(spaces, 'repo_type') and spaces.repo_type == "space"
    assert hasattr(papers, 'repo_type') and papers.repo_type == "paper"

    # Test that they have the required methods
    for singleton in [datasets, models, spaces, papers]:
        assert hasattr(singleton, 'search')
        assert hasattr(singleton, 'get_size')
        assert hasattr(singleton, '__getitem__')
        assert hasattr(singleton, '__iter__')
        assert hasattr(singleton, '__len__')
        assert hasattr(singleton, '__contains__')

    # Test that they are different instances
    assert datasets is not models
    assert models is not spaces
    assert spaces is not papers

    # Test that we can use them immediately without instantiation
    assert len(datasets) >= 0  # Should work immediately
    assert len(models) >= 0  # Should work immediately


def test_parameterized_hf_mapping():
    """Test that HfMapping can be used with direct parameterization."""
    from hfdol.base import RepoType, HfMapping

    # Test with enum values
    dataset_mapping = HfMapping(RepoType.DATASET)
    assert dataset_mapping.repo_type == "dataset"

    model_mapping = HfMapping(RepoType.MODEL)
    assert model_mapping.repo_type == "model"

    # Test with string values (should work due to str, Enum)
    space_mapping = HfMapping("space")
    assert space_mapping.repo_type == "space"

    paper_mapping = HfMapping("paper")
    assert paper_mapping.repo_type == "paper"

    # Test that parameterized mappings have the correct functions
    for mapping in [dataset_mapping, model_mapping, space_mapping, paper_mapping]:
        assert hasattr(mapping, 'loader_func')
        assert hasattr(mapping, 'search_func')
        assert mapping.loader_func is not None
        assert mapping.search_func is not None


def test_dynamic_search_signatures():
    """Test that search methods automatically have the correct signatures."""
    from inspect import signature
    from huggingface_hub import list_datasets, list_models

    # Get the signatures
    datasets_search_sig = signature(datasets.search)
    models_search_sig = signature(models.search)
    list_datasets_sig = signature(list_datasets)
    list_models_sig = signature(list_models)

    # The parameters should match (excluding 'self' for the bound methods)
    datasets_params = list(datasets_search_sig.parameters.keys())
    list_datasets_params = list(list_datasets_sig.parameters.keys())

    models_params = list(models_search_sig.parameters.keys())
    list_models_params = list(list_models_sig.parameters.keys())

    # Compare parameter names (datasets.search should have same params as list_datasets)
    assert (
        datasets_params == list_datasets_params
    ), f"datasets.search params {datasets_params} don't match list_datasets params {list_datasets_params}"
    assert (
        models_params == list_models_params
    ), f"models.search params {models_params} don't match list_models params {list_models_params}"

    # Test that the methods work with their native signatures
    try:
        # These should work because signatures are correctly aligned
        datasets_results = list(datasets.search(filter='tiny', limit=2))
        models_results = list(models.search(filter='test', limit=2))

        assert len(datasets_results) >= 0
        assert len(models_results) >= 0
    except Exception as e:
        pytest.fail(f"Dynamic signature search methods failed: {e}")
