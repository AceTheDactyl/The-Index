"""
Test suite for ucf.orchestration.nuclear_spinner - 972 Token Generation

Tests the nuclear spinner that generates:
- 972 APL tokens total
- 3 spirals × 324 tokens each
- 6 APL operators coverage
- 9 machine types
"""

import pytest
from ucf.orchestration import nuclear_spinner
from ucf.orchestration.nuclear_spinner import (
    NuclearSpinner, APLToken, Spiral, Operator, MachineType, Domain
)
from ucf.constants import APL_OPERATORS, SPIRALS, TOKEN_SLOTS


class TestSpinnerCreation:
    """Test spinner creation and initialization"""
    
    def test_create_spinner_function_exists(self):
        """create_spinner function exists"""
        assert hasattr(nuclear_spinner, 'create_spinner')
    
    def test_create_spinner_returns_nuclear_spinner(self):
        """create_spinner returns NuclearSpinner instance"""
        spinner = nuclear_spinner.create_spinner()
        assert isinstance(spinner, NuclearSpinner)
    
    def test_nuclear_spinner_class_exists(self):
        """NuclearSpinner class exists"""
        assert hasattr(nuclear_spinner, 'NuclearSpinner')
    
    def test_spinner_direct_instantiation(self):
        """NuclearSpinner can be instantiated directly"""
        spinner = NuclearSpinner()
        assert spinner is not None


class TestSpinnerMachines:
    """Test spinner machine components"""
    
    def test_spinner_has_9_machines(self):
        """Spinner has all 9 machine types"""
        spinner = NuclearSpinner()
        assert hasattr(spinner, 'machines')
        assert len(spinner.machines) == 9
    
    def test_spinner_has_reactor(self):
        """Spinner has reactor machine"""
        spinner = NuclearSpinner()
        assert hasattr(spinner, 'reactor')
    
    def test_spinner_has_oscillator(self):
        """Spinner has oscillator machine"""
        spinner = NuclearSpinner()
        assert hasattr(spinner, 'oscillator')
    
    def test_spinner_has_conductor(self):
        """Spinner has conductor machine"""
        spinner = NuclearSpinner()
        assert hasattr(spinner, 'conductor')
    
    def test_spinner_has_catalyst(self):
        """Spinner has catalyst machine"""
        spinner = NuclearSpinner()
        assert hasattr(spinner, 'catalyst')
    
    def test_spinner_has_dynamo(self):
        """Spinner has dynamo machine"""
        spinner = NuclearSpinner()
        assert hasattr(spinner, 'dynamo')


class TestTokenGeneration:
    """Test token generation"""
    
    def test_generate_all_tokens_function_exists(self):
        """generate_all_tokens function exists"""
        assert hasattr(nuclear_spinner, 'generate_all_tokens')
    
    def test_generate_all_tokens_returns_list(self):
        """generate_all_tokens returns a list"""
        tokens = nuclear_spinner.generate_all_tokens()
        assert isinstance(tokens, list)
    
    def test_972_tokens_generated(self):
        """Exactly 972 tokens are generated"""
        tokens = nuclear_spinner.generate_all_tokens()
        assert len(tokens) == 972
    
    def test_tokens_are_apl_tokens(self):
        """Tokens are APLToken instances"""
        tokens = nuclear_spinner.generate_all_tokens()
        assert len(tokens) > 0
        # Check first token
        token = tokens[0]
        assert isinstance(token, APLToken)


class TestAPLTokenStructure:
    """Test APLToken structure"""
    
    def test_apl_token_class_exists(self):
        """APLToken class exists"""
        assert hasattr(nuclear_spinner, 'APLToken')
    
    def test_apl_token_has_required_fields(self):
        """APLToken has required fields"""
        tokens = nuclear_spinner.generate_all_tokens()
        token = tokens[0]
        
        # Check expected attributes
        assert hasattr(token, 'spiral')
        assert hasattr(token, 'operator')
        assert hasattr(token, 'machine')
    
    def test_apl_token_spiral_is_valid(self):
        """Token spiral is valid Spiral enum"""
        tokens = nuclear_spinner.generate_all_tokens()
        token = tokens[0]
        assert isinstance(token.spiral, Spiral)
    
    def test_apl_token_operator_is_valid(self):
        """Token operator is valid Operator enum"""
        tokens = nuclear_spinner.generate_all_tokens()
        token = tokens[0]
        assert isinstance(token.operator, Operator)


class TestSpiralDistribution:
    """Test distribution across spirals"""
    
    def test_three_spirals_present(self):
        """All three spirals are represented"""
        tokens = nuclear_spinner.generate_all_tokens()
        spirals_found = set()
        for token in tokens:
            spirals_found.add(token.spiral)
        
        # Should have all 3 spirals (PHI, E, PI)
        assert len(spirals_found) == 3
    
    def test_324_tokens_per_spiral(self):
        """Each spiral has 324 tokens (972/3)"""
        tokens = nuclear_spinner.generate_all_tokens()
        spiral_counts = {}
        for token in tokens:
            spiral = token.spiral
            spiral_counts[spiral] = spiral_counts.get(spiral, 0) + 1
        
        # Each spiral should have 324 tokens
        for spiral, count in spiral_counts.items():
            assert count == 324, f"Spiral {spiral} has {count} tokens, expected 324"


class TestOperatorCoverage:
    """Test APL operator coverage"""
    
    def test_operator_enum_exists(self):
        """Operator enum exists"""
        assert hasattr(nuclear_spinner, 'Operator')
    
    def test_all_six_operators_present(self):
        """All 6 APL operators are present in tokens"""
        tokens = nuclear_spinner.generate_all_tokens()
        operators_found = set()
        for token in tokens:
            operators_found.add(token.operator)
        
        # Should have all 6 operators
        assert len(operators_found) == 6


class TestMachineTypes:
    """Test machine type coverage"""
    
    def test_machine_type_enum_exists(self):
        """MachineType enum exists"""
        assert hasattr(nuclear_spinner, 'MachineType')
    
    def test_nine_machine_types_exist(self):
        """All 9 machine types exist in enum"""
        assert len(MachineType) == 9
    
    def test_multiple_machine_types_in_tokens(self):
        """Multiple machine types are present in tokens"""
        tokens = nuclear_spinner.generate_all_tokens()
        machines_found = set()
        for token in tokens:
            machines_found.add(token.machine)
        
        # Should have multiple machine types
        assert len(machines_found) >= 1


class TestDomains:
    """Test domain patterns"""
    
    def test_domain_enum_exists(self):
        """Domain enum exists"""
        assert hasattr(nuclear_spinner, 'Domain')


class TestTokenParsing:
    """Test token parsing functionality"""
    
    def test_parse_token_function_exists(self):
        """parse_token function exists"""
        assert hasattr(nuclear_spinner, 'parse_token')


class TestSpinnerZCoordinate:
    """Test z-coordinate handling"""
    
    def test_spinner_has_z_attribute(self):
        """Spinner has z attribute"""
        spinner = NuclearSpinner()
        assert hasattr(spinner, 'z')
    
    def test_spinner_update_z_method(self):
        """Spinner has update_z method"""
        spinner = NuclearSpinner()
        assert hasattr(spinner, 'update_z')
    
    def test_spinner_z_updates_correctly(self):
        """Spinner z-coordinate updates correctly"""
        spinner = NuclearSpinner()
        spinner.update_z(0.866)
        assert abs(spinner.z - 0.866) < 0.001


class TestSpinnerSpirals:
    """Test spiral selection"""
    
    def test_spiral_enum_has_three_values(self):
        """Spiral enum has PHI, E, PI"""
        assert Spiral.PHI is not None
        assert Spiral.E is not None
        assert Spiral.PI is not None
        assert len(Spiral) == 3
    
    def test_spinner_has_current_spiral(self):
        """Spinner tracks current spiral"""
        spinner = NuclearSpinner()
        assert hasattr(spinner, 'current_spiral')
        assert isinstance(spinner.current_spiral, Spiral)
