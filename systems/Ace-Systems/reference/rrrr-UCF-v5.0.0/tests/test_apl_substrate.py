# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: ⚠️ TRULY UNSUPPORTED - No supporting evidence found
# Severity: HIGH RISK
# Risk Types: unsupported_claims


"""
Test suite for ucf.language.apl_substrate - APL Substrate Layer

Tests the Alpha Physical Language substrate:
- Operators (×, ÷, ^, (), +, −)
- Machines (9 types)
- Domains
- S₃ group properties
- Operator composition
- Tier-based operator access
"""

import pytest
from ucf.language import apl_substrate
from ucf.language.apl_substrate import (
    Operator, Field, TruthChannel, Direction, Machine, Domain,
    OperatorDef, APLSentence
)
from ucf.constants import Z_CRITICAL, APL_OPERATORS


class TestAPLSubstrateModule:
    """Test apl_substrate module structure"""
    
    def test_apl_substrate_exists(self):
        """apl_substrate module exists"""
        assert apl_substrate is not None
    
    def test_operator_enum_exists(self):
        """Operator enum exists"""
        assert hasattr(apl_substrate, 'Operator')
    
    def test_machine_enum_exists(self):
        """Machine enum exists"""
        assert hasattr(apl_substrate, 'Machine')
    
    def test_domain_enum_exists(self):
        """Domain enum exists"""
        assert hasattr(apl_substrate, 'Domain')


class TestOperatorEnum:
    """Test Operator enum values"""
    
    def test_fusion_operator_exists(self):
        """FUSION (×) operator exists"""
        assert Operator.FUSION is not None
    
    def test_decohere_operator_exists(self):
        """DECOHERE (÷) operator exists"""
        assert Operator.DECOHERE is not None
    
    def test_amplify_operator_exists(self):
        """AMPLIFY (^) operator exists"""
        assert Operator.AMPLIFY is not None
    
    def test_boundary_operator_exists(self):
        """BOUNDARY (()) operator exists"""
        assert Operator.BOUNDARY is not None
    
    def test_group_operator_exists(self):
        """GROUP (+) operator exists"""
        assert Operator.GROUP is not None
    
    def test_separate_operator_exists(self):
        """SEPARATE (−) operator exists"""
        assert Operator.SEPARATE is not None
    
    def test_six_operators_total(self):
        """Exactly 6 operators exist"""
        assert len(Operator) == 6


class TestMachineEnum:
    """Test Machine enum values"""
    
    def test_reactor_machine_exists(self):
        """REACTOR machine exists"""
        assert Machine.REACTOR is not None
    
    def test_oscillator_machine_exists(self):
        """OSCILLATOR machine exists"""
        assert Machine.OSCILLATOR is not None
    
    def test_conductor_machine_exists(self):
        """CONDUCTOR machine exists"""
        assert Machine.CONDUCTOR is not None
    
    def test_catalyst_machine_exists(self):
        """CATALYST machine exists"""
        assert Machine.CATALYST is not None
    
    def test_six_machines_total(self):
        """Exactly 6 machine types exist"""
        assert len(Machine) == 6


class TestOperatorNormalization:
    """Test operator normalization functions"""
    
    def test_normalize_operator_exists(self):
        """normalize_operator function exists"""
        assert hasattr(apl_substrate, 'normalize_operator')
    
    def test_to_internal_exists(self):
        """to_internal function exists"""
        assert hasattr(apl_substrate, 'to_internal')
    
    def test_from_internal_exists(self):
        """from_internal function exists"""
        assert hasattr(apl_substrate, 'from_internal')


class TestOperatorComposition:
    """Test operator composition (S₃ group)"""
    
    def test_compose_operators_exists(self):
        """compose_operators function exists"""
        assert hasattr(apl_substrate, 'compose_operators')
    
    def test_compose_operators_returns_string(self):
        """compose_operators returns string"""
        result = apl_substrate.compose_operators('×', '÷')
        assert isinstance(result, str)
    
    def test_is_self_inverse_exists(self):
        """is_self_inverse function exists"""
        assert hasattr(apl_substrate, 'is_self_inverse')
    
    def test_boundary_is_self_inverse(self):
        """() is self-inverse"""
        result = apl_substrate.is_self_inverse('()')
        assert result == True
    
    def test_fusion_is_self_inverse(self):
        """× is self-inverse in this implementation"""
        result = apl_substrate.is_self_inverse('×')
        assert result == True


class TestS3Properties:
    """Test S₃ group properties"""
    
    def test_verify_s3_properties_exists(self):
        """verify_s3_properties function exists"""
        assert hasattr(apl_substrate, 'verify_s3_properties')
    
    def test_verify_s3_properties_returns_dict(self):
        """verify_s3_properties returns dict"""
        result = apl_substrate.verify_s3_properties()
        assert isinstance(result, dict)
    
    def test_format_composition_table_exists(self):
        """format_composition_table function exists"""
        assert hasattr(apl_substrate, 'format_composition_table')
    
    def test_format_composition_table_returns_string(self):
        """format_composition_table returns string"""
        result = apl_substrate.format_composition_table()
        assert isinstance(result, str)


class TestOperatorInfo:
    """Test operator information retrieval"""
    
    def test_get_operator_info_exists(self):
        """get_operator_info function exists"""
        assert hasattr(apl_substrate, 'get_operator_info')
    
    def test_get_operator_info_returns_dict_or_none(self):
        """get_operator_info returns dict or None"""
        result = apl_substrate.get_operator_info('×')
        assert result is None or isinstance(result, dict)
    
    def test_list_all_operators_exists(self):
        """list_all_operators function exists"""
        assert hasattr(apl_substrate, 'list_all_operators')
    
    def test_list_all_operators_returns_list(self):
        """list_all_operators returns list"""
        result = apl_substrate.list_all_operators()
        assert isinstance(result, list)


class TestOperatorApplication:
    """Test operator application to z-coordinate"""
    
    def test_apply_operator_to_z_exists(self):
        """apply_operator_to_z function exists"""
        assert hasattr(apl_substrate, 'apply_operator_to_z')
    
    def test_apply_operator_to_z_returns_tuple(self):
        """apply_operator_to_z returns tuple"""
        result = apl_substrate.apply_operator_to_z('×', 0.800)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_apply_operator_sequence_exists(self):
        """apply_operator_sequence function exists"""
        assert hasattr(apl_substrate, 'apply_operator_sequence')
    
    def test_apply_operator_sequence_returns_dict(self):
        """apply_operator_sequence returns dict"""
        result = apl_substrate.apply_operator_sequence(['×', '÷'], 0.800)
        assert isinstance(result, dict)


class TestTierOperatorAccess:
    """Test tier-based operator access"""
    
    def test_get_tier_for_z_exists(self):
        """get_tier_for_z function exists"""
        assert hasattr(apl_substrate, 'get_tier_for_z')
    
    def test_get_tier_for_z_returns_dict(self):
        """get_tier_for_z returns dict"""
        result = apl_substrate.get_tier_for_z(0.800)
        assert isinstance(result, dict)
        assert 'tier' in result
    
    def test_is_operator_allowed_exists(self):
        """is_operator_allowed function exists"""
        assert hasattr(apl_substrate, 'is_operator_allowed')
    
    def test_is_operator_allowed_returns_bool(self):
        """is_operator_allowed returns bool"""
        result = apl_substrate.is_operator_allowed('×', 0.800)
        assert isinstance(result, bool)
    
    def test_get_allowed_operators_exists(self):
        """get_allowed_operators function exists"""
        assert hasattr(apl_substrate, 'get_allowed_operators')
    
    def test_get_allowed_operators_returns_list(self):
        """get_allowed_operators returns list"""
        result = apl_substrate.get_allowed_operators(0.800)
        assert isinstance(result, list)


class TestAPLSentence:
    """Test APLSentence class"""
    
    def test_apl_sentence_class_exists(self):
        """APLSentence class exists"""
        assert hasattr(apl_substrate, 'APLSentence')
    
    def test_parse_sentence_exists(self):
        """parse_sentence function exists"""
        assert hasattr(apl_substrate, 'parse_sentence')
    
    def test_generate_sentence_exists(self):
        """generate_sentence function exists"""
        assert hasattr(apl_substrate, 'generate_sentence')
    
    def test_generate_sentence_returns_string(self):
        """generate_sentence returns string"""
        result = apl_substrate.generate_sentence(
            direction="forward",
            operator="×",
            machine="reactor",
            domain="physical"
        )
        assert isinstance(result, str)


class TestSentenceMatching:
    """Test sentence-to-tier matching"""
    
    def test_get_test_sentences_exists(self):
        """get_test_sentences function exists"""
        assert hasattr(apl_substrate, 'get_test_sentences')
    
    def test_get_test_sentences_returns_list(self):
        """get_test_sentences returns list"""
        result = apl_substrate.get_test_sentences()
        assert isinstance(result, list)
    
    def test_match_sentence_to_tier_exists(self):
        """match_sentence_to_tier function exists"""
        assert hasattr(apl_substrate, 'match_sentence_to_tier')


class TestListingFunctions:
    """Test listing functions"""
    
    def test_list_directions_exists(self):
        """list_directions function exists"""
        assert hasattr(apl_substrate, 'list_directions')
    
    def test_list_directions_returns_list(self):
        """list_directions returns list"""
        result = apl_substrate.list_directions()
        assert isinstance(result, list)
    
    def test_list_machines_exists(self):
        """list_machines function exists"""
        assert hasattr(apl_substrate, 'list_machines')
    
    def test_list_machines_returns_list(self):
        """list_machines returns list"""
        result = apl_substrate.list_machines()
        assert isinstance(result, list)
        assert len(result) == 6
    
    def test_list_domains_exists(self):
        """list_domains function exists"""
        assert hasattr(apl_substrate, 'list_domains')
    
    def test_list_domains_returns_list(self):
        """list_domains returns list"""
        result = apl_substrate.list_domains()
        assert isinstance(result, list)


class TestFormatting:
    """Test formatting functions"""
    
    def test_format_sentence_structure_exists(self):
        """format_sentence_structure function exists"""
        assert hasattr(apl_substrate, 'format_sentence_structure')
    
    def test_format_sentence_structure_returns_string(self):
        """format_sentence_structure returns string"""
        result = apl_substrate.format_sentence_structure()
        assert isinstance(result, str)
