# VisProbe API Architecture Overview

This document provides an overview of the refactored `visprobe.api` module, designed for clarity, simplicity, and maintainability.

## Core Concepts

The API is built around two key concepts:

1.  **Declarative Decorators**: Users define tests using a simple, clear set of decorators (`@given`, `@search`, `@model`, `@data_source`). These decorators are lightweight and only serve to configure the test.

2.  **Orchestrated Execution**: `TestRunner` class handles the entire lifecycle of a test run. It takes the configuration from the decorators and performs all the complex tasks of execution, analysis, and reporting.

## Directory Structure

-   `src/visprobe/api/`
    -   `__init__.py`: Exports the public API components.
    -   `decorators.py`: Contains the simple, user-facing decorators.
    -   `runner.py`: Home of the `TestRunner` class, the engine of the framework.
    -   `config.py`: Centralized configuration and constants.
    -   `report.py`: Defines the unified `Report` data structure for all test results.
    -   `registry.py`: Manages the discovery of tests for the CLI.

## How It Works: A Step-by-Step Flow

1.  **Test Definition** (User's Code):
    A user defines a test using the decorators:
    ```python
    @search(strategy_factory=..., ...)
    @model(my_model, capture_intermediate_layers=[...])
    @data_source(my_data, collate_fn=...)
    def my_robustness_test(original, perturbed):
        assert my_property(original, perturbed)
    ```

2.  **Decorator Execution** (`decorators.py`):
    - When `my_robustness_test()` is called, the `@search` decorator instantiates the `TestRunner`, passing it the user's test function, the test type (`'search'`), and all its parameters.
    - `runner = TestRunner(user_func, 'search', kwargs)`

3.  **Test Execution** (`runner.py`):
    - The `TestRunner`'s `run()` method is called.
    - **Context Gathering**: The runner inspects the user's test function to get the model and data configuration attached by the `@model` and `@data_source` decorators.
    - **Model Wrapping**: If `capture_intermediate_layers` was provided, the runner wraps the model with the internal `_ModelWithIntermediateOutput` utility.
    - **Core Loop**: The runner executes the appropriate test loop (`_run_search()` or `_run_given()`).
    - **Analysis**: After the core loop, the runner executes any additional analyses (noise sensitivity, resolution impact, etc.).
    - **Report Building**: The runner gathers all results into a single, unified `Report` object.
    - **Saving**: The final report is saved to a JSON file.
    - The `Report` object is returned to the user.

## Benefits of This Architecture

-   **Simple and Clean Decorators**: The user-facing API is easy to understand.
-   **Centralized Logic**: The `TestRunner` provides a single place for all orchestration logic.
-   **Extensible**: New features can be added by modifying the `TestRunner` without complicating the decorators.
-   **Unified Reporting**: A single `Report` class handles all test result data, simplifying the codebase.
-   **Maintainable**: The clear separation of concerns makes the code easier to reason about and modify.
