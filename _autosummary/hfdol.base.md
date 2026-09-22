# hfdol.base

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

### Functions

| `ensure_dir`(dirpath)                                                                             |                                                                                           |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `ensure_id`(obj)                                                                                  |                                                                                           |
| [`get_size`](#hfdol.base.get_size)(repo_id, \*[, unit_bytes])              | Calculates the total size of a Hugging Face repository (model, dataset, or space) in GiB. |
| [`list_local_repos`](#hfdol.base.list_local_repos)(repo_type)                      | Dynamically list locally cached repositories of a given type using scan_cache_dir.        |
| [`sign_kwargs_with`](#hfdol.base.sign_kwargs_with)(source_func[, first_arg_index]) | Minimal decorator to replace \*\*kwargs with parameters from source_func.                 |

### Classes

| [`HfDatasets`](#hfdol.base.HfDatasets)([repo_type])   | A Mapping interface to HuggingFace datasets.   |
|----------------------------------------------------------------------------|------------------------------------------------|
| [`HfMapping`](#hfdol.base.HfMapping)([repo_type])    | Abstract base class for HuggingFace Mappings.  |
| [`HfModels`](#hfdol.base.HfModels)([repo_type])     | A Mapping interface to HuggingFace models.     |
| [`HfPapers`](#hfdol.base.HfPapers)([repo_type])     | A Mapping interface to HuggingFace Papers.     |
| [`HfSpaces`](#hfdol.base.HfSpaces)([repo_type])     | A Mapping interface to HuggingFace Spaces.     |
| [`RepoType`](#hfdol.base.RepoType)(\*values)        | Valid HuggingFace repository types.            |

### *class* hfdol.base.HfDatasets(repo_type=None)

Bases: [`HfMapping`](#hfdol.base.HfMapping)

A Mapping interface to HuggingFace datasets.

Provides dictionary-like access to locally cached datasets and seamless
downloading of remote datasets. Keys are dataset repository IDs (e.g., ‘stingning/ultrachat’).
Values are loaded dataset objects from the datasets library.

### Examples

```pycon
>>> d = HfDatasets()
```

List locally cached datasets:

```pycon
>>> list_of_dataset_repo_ids = list(d)
```

Search remote datasets

```pycon
>>> results = d.search('music', gated=False)
```

Load or download a dataset

```pycon
>>> data = d['some/dataset']
```

### *class* hfdol.base.HfMapping(repo_type=None)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Abstract base class for HuggingFace Mappings.

This class provides common functionality for datasets, models, spaces, and papers,
parameterized by repo_type which determines loader_func and search_func.

Can be used in two ways:

1. Via subclasses (HfDatasets, HfModels, etc.) for common types - best UX
2. Via direct parameterization for less common types or dynamic use

#### get_size(key, , unit_bytes=1073741824)

Get size (by default, in GiB) of an item from it’s key (repo ID)

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### *class* hfdol.base.HfModels(repo_type=None)

Bases: [`HfMapping`](#hfdol.base.HfMapping)

A Mapping interface to HuggingFace models.

Provides dictionary-like access to locally cached models and seamless
downloading of remote models. Keys are model repository IDs (e.g., ‘sentence-transformers/all-MiniLM-L6-v2’).
Values are file paths to the downloaded model directories.

### Examples

```pycon
>>> m = HfModels()
```

List locally cached models

```pycon
>>> list_of_model_repo_ids = list(m)
```

Search remote models

```pycon
>>> results = m.search('embeddings', gated=False)
```

Download or get path to a model

```pycon
>>> model_path = m['some/model']
```

### *class* hfdol.base.HfPapers(repo_type=None)

Bases: [`HfMapping`](#hfdol.base.HfMapping)

A Mapping interface to HuggingFace Papers.

Provides dictionary-like access to paper information. Keys are paper IDs.
Values are PaperInfo objects containing paper metadata, abstracts, and links.

#### NOTE
Papers are metadata objects only - they don’t have downloadable files or sizes.

### Examples

```pycon
>>> p = HfPapers()
```

Search papers

```pycon
>>> results = p.search('transformer')
```

Get information about a paper

```pycon
>>> paper_info = p['2017.12345']
```

### *class* hfdol.base.HfSpaces(repo_type=None)

Bases: [`HfMapping`](#hfdol.base.HfMapping)

A Mapping interface to HuggingFace Spaces.

Provides dictionary-like access to locally cached spaces and seamless
retrieval of remote space information. Keys are space repository IDs (e.g., ‘gradio/chatbot’).
Values are SpaceInfo objects containing space metadata and configuration.

### Examples

```pycon
>>> s = HfSpaces()
```

List locally cached spaces

```pycon
>>> list_of_space_repo_ids = list(s)
```

Search remote spaces

```pycon
>>> results = s.search('gradio', gated=False)
```

Get information about a space

```pycon
>>> space_info = s['gradio/chatbot']
```

### *class* hfdol.base.RepoType(\*values)

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Valid HuggingFace repository types.

### hfdol.base.get_size(repo_id, , unit_bytes=1073741824, repo_type)

Calculates the total size of a Hugging Face repository (model, dataset, or space) in GiB.

* **Parameters:**
  * **repo_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The ID of the repository (e.g., “bert-base-uncased”, “MMMU/MMMU”, or “spaces/gradio/chatbot”).
  * **unit_bytes** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of bytes in the desired unit. Default is 1024\*\*3 for GiB. For bytes, enter 1.
  * **repo_type** ([`RepoType`](#hfdol.base.RepoType)) – Type of repository (“model”, “dataset”, “space”). Required parameter.
    Papers don’t have file sizes, so they’re not supported.
* **Returns:**
  The total repository size in the specified unit.
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If repo_type is “paper” (papers don’t have file sizes) or if repo_type is invalid.

### hfdol.base.list_local_repos(repo_type)

Dynamically list locally cached repositories of a given type using scan_cache_dir.

* **Parameters:**
  **repo_type** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The type of repository (“dataset”, “model”, “space”, or “paper”).
* **Returns:**
  A list of repo IDs for repositories of the specified type.
* **Return type:**
  [*list*](https://docs.python.org/3/builtins/stdtypes.html#list)

### hfdol.base.sign_kwargs_with(source_func, first_arg_index=0)

Minimal decorator to replace \*\*kwargs with parameters from source_func.

* **Parameters:**
  * **source_func** – Function whose parameters will be injected
  * **first_arg_index** – Index of first parameter to include from source (default 0)

```pycon
>>> def source(x=0, y=1, z=2): ...
>>> @sign_kwargs_with(source)
... def target(some, thing=None, **kwargs):
...     pass
>>> str(signature(target))
'(some, thing=None, *, x=0, y=1, z=2)'
```

```pycon
>>> @sign_kwargs_with(source, first_arg_index=1)
... def target2(some, thing=None, **kwargs):
...     pass
>>> str(signature(target2))
'(some, thing=None, *, y=1, z=2)'
```
