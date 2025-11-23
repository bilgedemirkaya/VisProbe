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
        TestRegistry.clear_all()

        @given(strategy=GaussianNoiseStrategy(std_dev=0.1), property_fn=LabelConstant())
        def my_test(model_fn, data_fn):
            return simple_model, sample_batch[0]

        assert len(TestRegistry.get_given_tests()) == 1

    def test_given_validates_parameters(self):
        """Test that @given validates function signature."""
        from visprobe.api.decorators import ValidationError

        with pytest.raises(ValidationError, match="must accept at least 2 parameters"):

            @given(strategy=GaussianNoiseStrategy(), property_fn=LabelConstant())
            def invalid_test(only_one_param):
                pass


class TestModelDecorator:
    """Tests for @model decorator."""

    def test_model_registers(self, simple_model):
        """Test that @model registers a model provider."""
        TestRegistry.clear_all()

        @model
        def my_model():
            return simple_model

        assert len(TestRegistry.get_model_providers()) == 1

    def test_model_validates_signature(self):
        """Test that @model validates function signature."""
        from visprobe.api.decorators import ValidationError

        with pytest.raises(ValidationError, match="must accept exactly 0 parameters"):

            @model
            def invalid_model(unexpected_param):
                pass


class TestDataSourceDecorator:
    """Tests for @data_source decorator."""

    def test_data_source_registers(self, sample_batch):
        """Test that @data_source registers a data provider."""
        TestRegistry.clear_all()

        @data_source
        def my_data():
            return sample_batch[0]

        assert len(TestRegistry.get_data_providers()) == 1

    def test_data_source_validates_signature(self):
        """Test that @data_source validates function signature."""
        from visprobe.api.decorators import ValidationError

        with pytest.raises(ValidationError, match="must accept exactly 0 parameters"):

            @data_source
            def invalid_data(unexpected_param):
                pass


class TestSearchDecorator:
    """Tests for @search decorator."""

    def test_search_registers_test(self, simple_model, sample_batch):
        """Test that @search registers a search test."""
        TestRegistry.clear_all()

        @search(
            strategy=GaussianNoiseStrategy(std_dev=0.1),
            property_fn=LabelConstant(),
            param_name="std_dev",
            param_min=0.0,
            param_max=1.0,
        )
        def my_search(model_fn, data_fn):
            return simple_model, sample_batch[0]

        assert len(TestRegistry.get_search_tests()) == 1

    def test_search_validates_parameters(self):
        """Test that @search validates function signature."""
        from visprobe.api.decorators import ValidationError

        with pytest.raises(ValidationError, match="must accept at least 2 parameters"):

            @search(
                strategy=GaussianNoiseStrategy(),
                property_fn=LabelConstant(),
                param_name="std_dev",
                param_min=0.0,
                param_max=1.0,
            )
            def invalid_test(only_one_param):
                pass


class TestRegistryIntegration:
    """Integration tests for decorator and registry interaction."""

    def test_multiple_tests_registered(self, simple_model, sample_batch):
        """Test that multiple tests can be registered."""
        TestRegistry.clear_all()

        @given(strategy=GaussianNoiseStrategy(std_dev=0.1), property_fn=LabelConstant())
        def test1(model_fn, data_fn):
            return simple_model, sample_batch[0]

        @given(strategy=GaussianNoiseStrategy(std_dev=0.2), property_fn=LabelConstant())
        def test2(model_fn, data_fn):
            return simple_model, sample_batch[0]

        assert len(TestRegistry.get_given_tests()) == 2

    def test_registry_clear(self, simple_model, sample_batch):
        """Test that registry can be cleared."""
        TestRegistry.clear_all()

        @given(strategy=GaussianNoiseStrategy(std_dev=0.1), property_fn=LabelConstant())
        def test1(model_fn, data_fn):
            return simple_model, sample_batch[0]

        assert len(TestRegistry.get_given_tests()) == 1

        TestRegistry.clear_all()
        assert len(TestRegistry.get_given_tests()) == 0
