"""
Test suite for ucf.core.unified_state - Unified State Management

Tests cross-layer synchronization between:
- Helix coordinates
- APL state  
- K.I.R.A. state
- TRIAD state

FIXED: Uses correct nested dict structure (state['apl']['z'] instead of state['z'])
"""

import pytest
from ucf.core import unified_state
from ucf.core.unified_state import UnifiedState, HelixState, KiraState, APLState, HelixCoordinate
from ucf.constants import PHI, Z_CRITICAL, PHI_INV


class TestUnifiedStateInitialization:
    """Test unified state initialization"""
    
    def setup_method(self):
        unified_state.reset_unified_state()
    
    def test_initial_state_exists(self):
        """Unified state can be retrieved"""
        state = unified_state.get_unified_state()
        assert state is not None
        assert isinstance(state, UnifiedState)
    
    def test_get_state_returns_dict(self):
        """get_state returns a dict"""
        state = unified_state.get_state()
        assert isinstance(state, dict)
    
    def test_initial_state_has_required_keys(self):
        """Initial state dict has required keys"""
        state = unified_state.get_state()
        assert 'mode' in state
        assert 'helix' in state
        assert 'kira' in state
        assert 'apl' in state
        assert 'consistency' in state
    
    def test_reset_creates_fresh_state(self):
        """Reset creates fresh state"""
        # Modify state
        unified_state.set_z(0.900)
        
        # Reset
        result = unified_state.reset_unified_state()
        
        # Should return fresh UnifiedState
        assert isinstance(result, UnifiedState)


class TestZCoordinatePropagation:
    """Test z-coordinate propagation across layers"""
    
    def setup_method(self):
        unified_state.reset_unified_state()
    
    def test_set_z_updates_apl_layer(self):
        """set_z updates the APL z-coordinate"""
        unified_state.set_z(0.850)
        state = unified_state.get_state()
        assert abs(state['apl']['z'] - 0.850) < 0.001
    
    def test_set_z_updates_helix_layer(self):
        """set_z updates the Helix z-coordinate"""
        unified_state.set_z(0.850)
        state = unified_state.get_state()
        assert abs(state['helix']['z'] - 0.850) < 0.001
    
    def test_set_z_returns_dict(self):
        """set_z returns updated state dict"""
        result = unified_state.set_z(0.850)
        assert isinstance(result, dict)
    
    def test_z_critical_special_handling(self):
        """Z_CRITICAL (0.866) is THE LENS"""
        unified_state.set_z(Z_CRITICAL)
        state = unified_state.get_state()
        assert abs(state['apl']['z'] - Z_CRITICAL) < 0.001


class TestHelixStateAccess:
    """Test Helix state access"""
    
    def setup_method(self):
        unified_state.reset_unified_state()
    
    def test_get_helix_returns_dict(self):
        """get_helix returns helix state dict"""
        helix = unified_state.get_helix()
        assert isinstance(helix, dict)
    
    def test_helix_has_z(self):
        """Helix state includes z-coordinate"""
        unified_state.set_z(0.800)
        helix = unified_state.get_helix()
        assert 'z' in helix
    
    def test_helix_has_coordinate(self):
        """Helix state includes coordinate string"""
        helix = unified_state.get_helix()
        assert 'coordinate' in helix


class TestAPLStateAccess:
    """Test APL state access"""
    
    def setup_method(self):
        unified_state.reset_unified_state()
    
    def test_get_apl_returns_dict(self):
        """get_apl returns APL state dict"""
        apl = unified_state.get_apl()
        assert isinstance(apl, dict)
    
    def test_apl_has_z(self):
        """APL state includes z-coordinate"""
        unified_state.set_z(0.850)
        apl = unified_state.get_apl()
        assert 'z' in apl
        assert abs(apl['z'] - 0.850) < 0.001
    
    def test_apl_has_phase(self):
        """APL state includes phase"""
        apl = unified_state.get_apl()
        assert 'phase' in apl
    
    def test_apl_has_tier(self):
        """APL state includes tier"""
        apl = unified_state.get_apl()
        assert 'tier' in apl


class TestKiraStateAccess:
    """Test K.I.R.A. state access"""
    
    def setup_method(self):
        unified_state.reset_unified_state()
    
    def test_get_kira_returns_dict(self):
        """get_kira returns K.I.R.A. state dict"""
        kira = unified_state.get_kira()
        assert isinstance(kira, dict)
    
    def test_kira_has_state(self):
        """K.I.R.A. state includes state field"""
        kira = unified_state.get_kira()
        assert 'state' in kira
    
    def test_kira_has_rail(self):
        """K.I.R.A. state includes rail field"""
        kira = unified_state.get_kira()
        assert 'rail' in kira


class TestStateSynchronization:
    """Test synchronization across state layers"""
    
    def setup_method(self):
        unified_state.reset_unified_state()
    
    def test_state_consistency_after_z_change(self):
        """All layers consistent after z change"""
        unified_state.set_z(0.880)
        
        state = unified_state.get_state()
        helix = unified_state.get_helix()
        apl = unified_state.get_apl()
        
        # All z values should match
        assert abs(helix['z'] - 0.880) < 0.001
        assert abs(apl['z'] - 0.880) < 0.001
    
    def test_check_sync_returns_dict(self):
        """check_sync returns consistency dict"""
        result = unified_state.check_sync()
        assert isinstance(result, dict)
        assert 'consistent' in result
    
    def test_consistency_maintained(self):
        """Consistency is maintained after operations"""
        unified_state.set_z(0.850)
        state = unified_state.get_state()
        assert state['consistency']['consistent'] == True


class TestPhaseDetection:
    """Test phase detection based on z-coordinate"""
    
    def setup_method(self):
        unified_state.reset_unified_state()
    
    def test_untrue_phase(self):
        """z < φ⁻¹ is UNTRUE phase"""
        unified_state.set_z(0.500)
        apl = unified_state.get_apl()
        assert apl['phase'] == 'UNTRUE'
    
    def test_paradox_phase(self):
        """φ⁻¹ <= z < z_c is PARADOX phase"""
        unified_state.set_z(0.800)
        apl = unified_state.get_apl()
        assert apl['phase'] == 'PARADOX'
    
    def test_true_phase(self):
        """z >= z_c is TRUE phase (or HYPER_TRUE)"""
        unified_state.set_z(Z_CRITICAL + 0.001)
        apl = unified_state.get_apl()
        assert apl['phase'] in ['TRUE', 'HYPER_TRUE']


class TestUnifiedStateClass:
    """Test UnifiedState class and related classes"""
    
    def test_unified_state_class_exists(self):
        """UnifiedState class exists"""
        assert hasattr(unified_state, 'UnifiedState')
    
    def test_helix_coordinate_class_exists(self):
        """HelixCoordinate class exists"""
        assert hasattr(unified_state, 'HelixCoordinate')
    
    def test_helix_state_class_exists(self):
        """HelixState class exists"""
        assert hasattr(unified_state, 'HelixState')
    
    def test_kira_state_class_exists(self):
        """KiraState class exists"""
        assert hasattr(unified_state, 'KiraState')
    
    def test_apl_state_class_exists(self):
        """APLState class exists"""
        assert hasattr(unified_state, 'APLState')


class TestFormatStatus:
    """Test status formatting"""
    
    def setup_method(self):
        unified_state.reset_unified_state()
    
    def test_format_status_returns_string(self):
        """format_status returns a string"""
        status = unified_state.format_status()
        assert isinstance(status, str)
        assert len(status) > 0
    
    def test_format_status_includes_z(self):
        """Status string includes z-coordinate info"""
        unified_state.set_z(0.850)
        status = unified_state.format_status()
        assert '0.85' in status or 'z' in status.lower()
