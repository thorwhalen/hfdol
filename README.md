# hfdol

Simple Mapping interface to HuggingFace.

(Note -- was [hf](https://pypi.org/project/hf/0.0.14/) but realeased the name to Huggingface itself for their tool.)

To install:	```pip install hfdol```

You'll also need a Hugginface token. See [more about this here](https://huggingface.co/docs/huggingface_hub/en/quick-start).


## Motivation

The Python packages [`datasets`](https://github.com/huggingface/datasets) and [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) provide a remarkably clean, well-documented, and comprehensive API for accessing datasets, models, spaces, and papers hosted on [Hugging Face](https://huggingface.co).  
Yet, as elegant as these APIs are, they remain *their own language*. Every library—no matter how intuitive—inevitably carries its own conventions, abstractions, and domain-specific semantics. When working with one or two APIs, this diversity is harmless, even stimulating. But when juggling dozens or hundreds of them, the cognitive overhead accumulates.

Despite their differences, most APIs share a small set of universal primitives — *retrieve something by key, list what's available, check existence, store, update, delete*.  
In Python, these operations are embodied by the `Mapping` interface, the conceptual model behind dictionaries. It's a minimal, ubiquitous, and instantly recognizable abstraction.  

This package offers such a `Mapping`-based façade to Hugging Face datasets and models, allowing you to browse, query, and access them as if they were simple Python dictionaries. The goal isn't to replace the original API, but to provide a thin, ergonomic layer for the most common operations — so you can spend less time remembering syntax, and more time working with data.

## Examples

This package provides four ready-to-use singleton instances, each offering a dictionary-like interface to different types of HuggingFace resources:

```python
import hfdoldol
```

### Working with Datasets

The `hfdol.datasets` singleton provides a `Mapping` (i.e. read-only-dictionary-like) interface to HuggingFace datasets:

#### List Local Datasets

As with dictionaries, `hfdol.datasets` is an iterable. An iterable of keys. 
The keys are repository ids for those datasets you've downloaded. 
See what datasets you already have cached locally like this:

```python
list(hfdol.datasets)  # Lists locally cached datasets
# ['stingning/ultrachat', 'allenai/WildChat-1M', 'google-research-datasets/go_emotions']
```

#### Access Local Datasets

The values of `hfdol.datasets` are the `DatasetDict` 
(from Huggingface's `datasets` package) instances that give you access to the dataset.
If you already have the dataset downloaded locally, it will load it from there, 
if not it will download it, then give it to you (and it will be cached locally 
for the next time you access it). 

```python
data = hfdol.datasets['stingning/ultrachat']  # Loads the dataset
print(data)  # Shows dataset information and structure
```

#### Search for Remote Datasets

`hfdol.datasets` also offers a search functionality, so you can search "remote" 
repositories:

```python
# Search for music-related datasets
search_results = hfdol.datasets.search('music', gated=False)
print(f"search_results is a {type(search_results).__name__}")  # It's a generator

# Get the first result (it will be a `DatasetInfo` instance contain information on the dataset)
result = next(search_results)
print(f"Dataset ID: {result.id}")
print(f"Description: {result.description[:80]}...")

# Download and use it directly
data = hfdol.datasets[result]  # You can pass the DatasetInfo object directly
```

Note that the `gated=False` was to make sure you get models that you have access to. 
For more search options, see the [HuggingFace Hub documentation](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_datasets).

#### A useful recipe: Get a table of result infos

You can use this to get a dataframe of the first/next `n` results of the results iterable:

```py
def table_of_results(results, n=10):
    import itertools, operator, pandas as pd

    results_table = pd.DataFrame(  # make a table with
        map(
            operator.attrgetter('__dict__'),  # the attributes dicts
            itertools.islice(results, n),  # ... of the first 10 search results
        )
    )
    return results_table
```

Example:

```py
results_table = table_of_results(search_results)
results_table
```

                              id            author                                       sha ...
    0   Genius-Society/hoyoMusic    Genius-Society  4f7e5120c0e8e26213d4bb3b52bcce76e69dfce4 ...
    1      Genius-Society/emo163    Genius-Society  6b8c3526b66940ddaedf15602d01083d24eb370c ...
    2  ccmusic-database/acapella  ccmusic-database  4cb8a4d4cb58cc55f30cb8c7a180fee1b5576dc5 ...
    3    ccmusic-database/pianos  ccmusic-database  db2b3f74c4c989b4fbda4b309e6bc925bfd8f5d1 ...
    ...


### Working with Models

The `hfdol.models` singleton provides the same dictionary-like interface for models:

#### Search for Models

Find models by keywords:

```python
model_search_results = hfdol.models.search('embeddings', gated=False)
model_result = next(model_search_results)
print(f"Model: {model_result.id}")
```

#### Download Models

Get the local path to a model (downloads if not cached):

```python
model_path = hfdol.models[model_result]
print(f"Model downloaded to: {model_path}")
```

#### List Local Models

See what models you have cached:

```python
list(hfdol.models)  # Lists all locally cached models
```

### Working with Spaces

The `hfdol.spaces` singleton provides access to HuggingFace Spaces (interactive ML demos and applications):

#### Search for Spaces

Find interesting Spaces by keywords:

```python
space_search_results = hfdol.spaces.search('gradio', limit=5)
space_result = next(space_search_results)
print(f"Space: {space_result.id}")
```

#### Access Space Information

Get detailed information about a Space:

```python
space_info = hfdol.spaces[space_result]
print(f"Space info: {space_info}")
```

#### List Local Spaces

See what spaces you have cached locally:

```python
list(hfdol.spaces)  # Lists all locally cached spaces
```

### Working with Papers

The `hfdol.papers` singleton provides access to research papers hosted on HuggingFace:

#### Search for Papers

Find research papers by topic:

```python
paper_search_results = hfdol.papers.search('transformer', limit=5)
paper_result = next(paper_search_results)
print(f"Paper: {paper_result.id}")
```

#### Access Paper Information

Get detailed information about a paper:

```python
paper_info = hfdol.papers[paper_result]
print(f"Paper title: {paper_info.title}")
print(f"Abstract: {paper_info.summary[:100]}...")
```

Note: Papers are metadata objects only—they contain information about research papers but don't have downloadable files like datasets or models.

### Getting Repository Sizes

You can check the size of any repository before downloading using the `get_size` function. The `repo_type` parameter is required to avoid ambiguity when repositories exist as multiple types:

```python
from hfdol import get_size

# Get size of a dataset (specify repo_type explicitly)
dataset_size = get_size('ccmusic-database/music_genre', repo_type='dataset')
print(f"Dataset size: {dataset_size:.2f} GiB")

# Get size of a model 
model_size = get_size('ccmusic-database/music_genre', repo_type='model')
print(f"Model size: {model_size:.2f} GiB")

# Using RepoType enum for type safety
from hfdol.base import RepoType
size_with_enum = get_size('some-repo', repo_type=RepoType.DATASET)

# Get size in different units (e.g., bytes)
size_in_bytes = get_size('some-repo', repo_type='dataset', unit_bytes=1)
```

**Pro tip**: Use the singleton instances for automatic repo_type handling:
```python
# These automatically know their repo_type
dataset_size = hfdol.datasets.get_size('ccmusic-database/music_genre')
model_size = hfdol.models.get_size('ccmusic-database/music_genre')
```

### Unified Interface

The beauty of this approach is that whether you're working with datasets, models, spaces, or papers, the interface remains familiar and consistent—just like working with Python dictionaries. All four singleton instances support the same core operations:

- **Dictionary-style access**: `resource = hfdol.datasets[key]`, `model_path = hfdol.models[key]`
- **Local listing**: `list(hfdol.datasets)`, `list(hfdol.models)` 
- **Remote searching**: `hfdol.datasets.search(query)`, `hfdol.models.search(query)`
- **Existence checking**: `key in hfdol.datasets`, `key in hfdol.models`

This unified interface means you can switch between different types of HuggingFace resources without learning new APIs—it's all just dictionaries! And since they're singleton instances, they're always ready to use without any setup.


## Design & Architecture

### Design Philosophy

This package is designed as a **thin façade** over the excellent [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) and [`datasets`](https://github.com/huggingface/datasets) libraries. Rather than reinventing functionality, it provides a unified `Mapping` interface that wraps the most common operations, making them feel like native Python dictionary operations.

The design balances two sometimes-competing goals:
1. **Simplicity**: Keep the codebase small, readable, and maintainable
2. **Single Source of Truth (SSOT)**: Minimize hardcoded knowledge about the underlying APIs

Ideally, this interface would be *entirely* auto-generated through static analysis of the wrapped packages. While we achieve this partially, practical constraints require some manual intervention—but we've minimized it as much as possible.

### Key Architectural Patterns

#### 1. Configuration-Driven Design (SSOT)

The `repo_type_helpers` dictionary serves as the **single source of truth** for all repo-type-specific behavior:

```python
repo_type_helpers = dict(
    dataset=dict(
        loader_func=load_dataset,
        search_func=list_datasets,
    ),
    model=dict(
        loader_func=snapshot_download,
        search_func=list_models,
    ),
    # ... etc
)
```

This declarative approach means:
- Adding a new repo type requires only updating this configuration
- No duplication of logic across different repo types
- Clear visibility of how each type differs

#### 2. Dynamic Signature Injection

Rather than manually replicating the signatures of wrapped functions (which would violate SSOT), we use **signature extraction and injection** via the `sign_kwargs_with` decorator:

```python
@sign_kwargs_with(search_func)
def search(self, filter, **kwargs):
    return self.search_func(filter=filter, **kwargs)
```

This means:
- Each `.search()` method automatically inherits the correct signature from its underlying function
- IDEs and type checkers see the actual parameters available
- When HuggingFace updates their APIs, our signatures update automatically
- Documentation stays accurate without manual synchronization

**Note**: The `list_papers` function required special handling (`_list_papers` wrapper) because it uses `query` instead of `filter` as its parameter name. This is the type of pragmatic compromise we make—we normalize the interface rather than exposing the inconsistency.

#### 3. Separation of Concerns

The architecture cleanly separates:

- **Configuration** (`repo_type_helpers`): What differs between types
- **Base functionality** (`HfMapping`): Shared behavior for all types
- **Type-specific classes** (`HfDatasets`, `HfModels`, etc.): Minimal subclasses that mainly provide:
  - Clear, discoverable class names
  - Type-specific documentation
  - Future extensibility points
- **Convenience layer** (module-level singletons): Zero-setup access for users

#### 4. Module-Level Singletons

The pre-instantiated `datasets`, `models`, `spaces`, and `papers` instances follow Python's **convenience instance pattern** (seen in `sys.stdout`, `np.random`, etc.):

```python
# Ready to use immediately
datasets = HfDatasets()
models = HfModels()
```

This works because these instances:
- Have no mutable state
- Require no configuration for basic use
- Represent logical singletons ("the datasets mapping")

#### 5. Progressive Disclosure

The API supports multiple levels of sophistication:

```python
# Simplest: Use pre-configured singletons
data = hfdol.datasets['some/dataset']

# Advanced: Create custom instances with configuration
my_datasets = HfDatasets()

# Power user: Parameterized mapping for dynamic repo types
custom = HfMapping(RepoType.DATASET)
```

### Design Compromises

Several compromises were made for pragmatism:

1. **Manual wrappers**: `_list_papers` normalizes the papers API to match others
2. **Enum + string hybrid**: `RepoType(str, Enum)` allows both type safety and string convenience
3. **Explicit repo_type in get_size**: Required parameter to avoid ambiguity when repos exist as multiple types
4. **Signature injection limitations**: Works well for keyword arguments but can't handle complex overloads

### Contributing Guidelines

When contributing to this package, please maintain these principles:

**✅ DO:**
- Add configuration to `repo_type_helpers` rather than creating new methods
- Use signature extraction (`sign_kwargs_with`) when wrapping functions with many parameters
- Keep `HfMapping` generic and push specialization to configuration
- Document *why* special cases exist (like `_list_papers`)
- Test against actual HuggingFace APIs to catch signature drift

**❌ AVOID:**
- Duplicating knowledge about wrapped APIs
- Hardcoding parameter lists or types that could be extracted
- Adding stateful behavior to mapping instances
- Creating wrapper methods that simply pass through to underlying functions

**When in doubt:**
- Ask "Could this be driven by configuration?"
- Prefer declarative patterns over imperative logic
- Keep the codebase small and the configuration visible

The goal is a package where 80% of the code is just wiring and configuration, and the HuggingFace packages do the actual work. This maximizes maintainability and minimizes drift as those packages evolve.
