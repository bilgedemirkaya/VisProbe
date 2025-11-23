"""
Tests for the visprobe.api.decorators module.
"""

import pytest

from visprobe.api.decorators import data_source, given, model, search
from visprobe.api.registry import TestRegistry
from visprobe.properties import LabelConstant
from visprobe.strategies.image import GaussianNoiseStrategy


class TestGivenDecorator:
    """Tests for @given decorator."""

    def test_given_registers_test(self, simple_model, sample_batch):
        """Test that @given registers a test function."""
        TestRegistry.clear()

        @given(strategy=GaussianNoiseStrategy(std_dev=0.1))
        def my_test(model_fn, data_fn):
            return simple_model, sample_batch[0]

        assert len(TestRegistry.get_given_tests()) == 1

    def test_given_validates_parameters(self):
        """Test that @given validates function signature."""
        from visprobe.api.decorators import ValidationError

        with pytest.raises(ValidationError, match="must accept at least 2 parameters"):

            @given(strategy=GaussianNoiseStrategy(std_dev=0.1))
            def invalid_test(only_one_param):
                pass


class TestModelDecorator:
    """Tests for @model decorator."""

    def test_model_attaches_metadata(self, simple_model):
        """Test that @model attaches model metadata to function."""

        @model(simple_model)
        def my_model():
            return simple_model

        assert hasattr(my_model, '_visprobe_model')
        assert my_model._visprobe_model is simple_model

    def test_model_with_intermediate_layers(self, simple_model):
        """Test that @model can capture intermediate layers."""

        @model(simple_model, capture_intermediate_layers=['layer1', 'layer2'])
        def my_model():
            return simple_model

        assert hasattr(my_model, '_visprobe_model')
        assert hasattr(my_model, '_visprobe_capture_intermediate_layers')
        assert my_model._visprobe_capture_intermediate_layers == ['layer1', 'layer2']


class TestDataSourceDecorator:
    """Tests for @data_source decorator."""

    def test_data_source_attaches_metadata(self, sample_batch):
        """Test that @data_source attaches data metadata to function."""

        @data_source(sample_batch[0])
        def my_data():
            return sample_batch[0]

        assert hasattr(my_data, '_visprobe_data')
        assert my_data._visprobe_data is sample_batch[0]

    def test_data_source_with_options(self, sample_batch):
        """Test that @data_source can include optional parameters."""

        def my_collate(x):
            return x

        @data_source(sample_batch[0], collate_fn=my_collate, class_names=['cat', 'dog'])
        def my_data():
            return sample_batch[0]

        assert hasattr(my_data, '_visprobe_data')
        assert hasattr(my_data, '_visprobe_collate')
        assert hasattr(my_data, '_visprobe_class_names')
        assert my_data._visprobe_collate is my_collate
        assert my_data._visprobe_class_names == ['cat', 'dog']


class TestSearchDecorator:
    """Tests for @search decorator."""

    def test_search_registers_test(self, simple_model, sample_batch):
        """Test that @search registers a search test."""
        TestRegistry.clear()

        @search(
            strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
            initial_level=0.01,
            level_lo=0.0,
            level_hi=1.0,
        )
        def my_search(original, perturbed):
            return True

        assert len(TestRegistry.get_search_tests()) == 1

    def test_search_validates_parameters(self):
        """Test that @search validates function signature."""
        from visprobe.api.decorators import ValidationError

        with pytest.raises(ValidationError, match="must accept at least 2 parameters"):

            @search(
                strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
                initial_level=0.01,
            )
            def invalid_test(only_one_param):
                pass


class TestRegistryIntegration:
    """Integration tests for decorator and registry interaction."""

    def test_multiple_tests_registered(self, simple_model, sample_batch):
        """Test that multiple tests can be registered."""
        TestRegistry.clear()

        @given(strategy=GaussianNoiseStrategy(std_dev=0.1))
        def test1(original, perturbed):
            return True

        @given(strategy=GaussianNoiseStrategy(std_dev=0.2))
        def test2(original, perturbed):
            return True

        assert len(TestRegistry.get_given_tests()) == 2

    def test_registry_clear(self, simple_model, sample_batch):
        """Test that registry can be cleared."""
        TestRegistry.clear()

        @given(strategy=GaussianNoiseStrategy(std_dev=0.1))
        def test1(original, perturbed):
            return True

        assert len(TestRegistry.get_given_tests()) == 1

        TestRegistry.clear()
        assert len(TestRegistry.get_given_tests()) == 0
