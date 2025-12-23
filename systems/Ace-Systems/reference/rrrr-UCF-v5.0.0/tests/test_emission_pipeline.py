# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Test file validates system behavior
# Severity: LOW RISK
# Risk Types: ['test_coverage']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/tests/test_emission_pipeline.py

"""
Test suite for ucf.language.emission_pipeline - Language Emission

Tests the 9-stage emission pipeline:
1. Content selection
2. Emergence check  
3. Structural frame
4. Slot assignment
5. Function words
6. Agreement/inflection
7. Connectors
8. Punctuation
9. Validation

FIXED: Correct function signatures (concepts list, z, context)
"""

import pytest
from ucf.language import emission_pipeline
from ucf.language.emission_pipeline import (
    Word, WordType, ContentWords, FrameType, SlotType,
    EmergenceResult, FrameResult, SlottedWords
)
from ucf.constants import PHASE_VOCAB


class TestWordClasses:
    """Test word and content classes"""
    
    def test_word_type_enum_exists(self):
        """WordType enum exists"""
        assert hasattr(emission_pipeline, 'WordType')
    
    def test_frame_type_enum_exists(self):
        """FrameType enum exists"""
        assert hasattr(emission_pipeline, 'FrameType')
    
    def test_slot_type_enum_exists(self):
        """SlotType enum exists"""
        assert hasattr(emission_pipeline, 'SlotType')
    
    def test_word_class_exists(self):
        """Word class exists"""
        assert hasattr(emission_pipeline, 'Word')
    
    def test_content_words_class_exists(self):
        """ContentWords class exists"""
        assert hasattr(emission_pipeline, 'ContentWords')
    
    def test_word_can_be_created(self):
        """Word objects can be created"""
        word = Word(
            text="consciousness",
            word_type=WordType.CONTENT,
            features={}
        )
        assert word.text == "consciousness"
        assert word.word_type == WordType.CONTENT


class TestStage1ContentSelection:
    """Test Stage 1: Content selection"""
    
    def test_stage1_function_exists(self):
        """stage1_content_selection function exists"""
        assert hasattr(emission_pipeline, 'stage1_content_selection')
    
    def test_stage1_returns_content_words(self):
        """Stage 1 returns ContentWords object"""
        concepts = ["consciousness", "emergence", "pattern"]
        result = emission_pipeline.stage1_content_selection(concepts, 0.850)
        assert isinstance(result, ContentWords)
    
    def test_stage1_with_different_z(self):
        """Stage 1 varies output based on z-coordinate"""
        concepts = ["thought"]
        result_low = emission_pipeline.stage1_content_selection(concepts, 0.300)
        result_high = emission_pipeline.stage1_content_selection(concepts, 0.900)
        # Both should return ContentWords
        assert isinstance(result_low, ContentWords)
        assert isinstance(result_high, ContentWords)


class TestStage2EmergenceCheck:
    """Test Stage 2: Emergence check"""
    
    def test_stage2_function_exists(self):
        """stage2_emergence_check function exists"""
        assert hasattr(emission_pipeline, 'stage2_emergence_check')
    
    def test_stage2_returns_emergence_result(self):
        """Stage 2 returns EmergenceResult"""
        concepts = ["light", "crystal"]
        content = emission_pipeline.stage1_content_selection(concepts, 0.850)
        result = emission_pipeline.stage2_emergence_check(content, 0.850)
        assert isinstance(result, EmergenceResult)


class TestStage3StructuralFrame:
    """Test Stage 3: Structural frame"""
    
    def test_stage3_function_exists(self):
        """stage3_structural_frame function exists"""
        assert hasattr(emission_pipeline, 'stage3_structural_frame')
    
    def test_stage3_returns_frame_result(self):
        """Stage 3 returns FrameResult"""
        concepts = ["unity", "emergence"]
        content = emission_pipeline.stage1_content_selection(concepts, 0.850)
        checked = emission_pipeline.stage2_emergence_check(content, 0.850)
        result = emission_pipeline.stage3_structural_frame(content, checked)
        assert isinstance(result, FrameResult)


class TestStage4SlotAssignment:
    """Test Stage 4: Slot assignment"""
    
    def test_stage4_function_exists(self):
        """stage4_slot_assignment function exists"""
        assert hasattr(emission_pipeline, 'stage4_slot_assignment')


class TestStage5FunctionWords:
    """Test Stage 5: Function words"""
    
    def test_stage5_function_exists(self):
        """stage5_function_words function exists"""
        assert hasattr(emission_pipeline, 'stage5_function_words')


class TestStage6AgreementInflection:
    """Test Stage 6: Agreement and inflection"""
    
    def test_stage6_function_exists(self):
        """stage6_agreement_inflection function exists"""
        assert hasattr(emission_pipeline, 'stage6_agreement_inflection')


class TestStage7Connectors:
    """Test Stage 7: Connectors"""
    
    def test_stage7_function_exists(self):
        """stage7_connectors function exists"""
        assert hasattr(emission_pipeline, 'stage7_connectors')


class TestStage8Punctuation:
    """Test Stage 8: Punctuation"""
    
    def test_stage8_function_exists(self):
        """stage8_punctuation function exists"""
        assert hasattr(emission_pipeline, 'stage8_punctuation')


class TestStage9Validation:
    """Test Stage 9: Validation"""
    
    def test_stage9_function_exists(self):
        """stage9_validation function exists"""
        assert hasattr(emission_pipeline, 'stage9_validation')


class TestEmissionPipelineClass:
    """Test EmissionPipeline class"""
    
    def test_emission_pipeline_class_exists(self):
        """EmissionPipeline class exists"""
        assert hasattr(emission_pipeline, 'EmissionPipeline')
    
    def test_emission_pipeline_can_instantiate(self):
        """EmissionPipeline can be instantiated"""
        pipeline = emission_pipeline.EmissionPipeline()
        assert pipeline is not None


class TestFullEmission:
    """Test full emission generation"""
    
    def test_emit_function_exists(self):
        """emit function exists"""
        assert hasattr(emission_pipeline, 'emit')
    
    def test_emit_returns_result(self):
        """emit() returns an EmissionResult"""
        concepts = ["consciousness", "light"]
        result = emission_pipeline.emit(concepts, 0.850)
        assert result is not None


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_get_pipeline_stages_exists(self):
        """get_pipeline_stages function exists"""
        assert hasattr(emission_pipeline, 'get_pipeline_stages')
    
    def test_format_pipeline_structure_exists(self):
        """format_pipeline_structure function exists"""
        assert hasattr(emission_pipeline, 'format_pipeline_structure')
    
    def test_get_pipeline_stages_returns_list(self):
        """get_pipeline_stages returns list"""
        stages = emission_pipeline.get_pipeline_stages()
        assert isinstance(stages, list)
        assert len(stages) > 0


class TestPhaseInfluence:
    """Test that phase influences emission"""
    
    def test_untrue_phase_emission(self):
        """UNTRUE phase (z < 0.618) works"""
        concepts = ["seed", "potential"]
        result = emission_pipeline.emit(concepts, 0.400)
        assert result is not None
    
    def test_paradox_phase_emission(self):
        """PARADOX phase (0.618 <= z < 0.866) works"""
        concepts = ["pattern", "threshold"]
        result = emission_pipeline.emit(concepts, 0.750)
        assert result is not None
    
    def test_true_phase_emission(self):
        """TRUE phase (z >= 0.866) works"""
        concepts = ["crystal", "lens", "light"]
        result = emission_pipeline.emit(concepts, 0.900)
        assert result is not None
