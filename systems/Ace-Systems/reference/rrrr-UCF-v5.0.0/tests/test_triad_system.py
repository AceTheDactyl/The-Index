"""
Test suite for ucf.core.triad_system - TRIAD Hysteresis FSM

Tests the critical TRIAD unlock mechanism:
- 3 crossings above 0.85 (rising) required
- Re-arm below 0.82 (falling) between crossings
- T6 gate locks/unlocks based on TRIAD state

FIXED: Uses object attributes (state.crossings) instead of dict subscripts (state["passes"])
"""

import pytest
from ucf.core import triad_system
from ucf.core.triad_system import TriadState, BandState, T6GateState
from ucf.constants import TRIAD_HIGH, TRIAD_LOW, TRIAD_PASSES_REQUIRED, Z_CRITICAL


class TestTriadStateInitialization:
    """Test TRIAD state initialization"""
    
    def setup_method(self):
        triad_system.reset_triad_state(0.800)
    
    def test_initial_state_locked(self):
        """TRIAD starts in LOCKED state"""
        state = triad_system.get_triad_state()
        assert state.unlocked == False
        assert state.crossings == 0
    
    def test_initial_z_stored(self):
        """Initial z-coordinate is stored"""
        state = triad_system.get_triad_state()
        assert state.z == 0.800
    
    def test_reset_clears_crossings(self):
        """Reset clears accumulated crossings"""
        # Accumulate some crossings
        triad_system.step(0.86)
        triad_system.step(0.81)
        triad_system.step(0.86)
        
        # Reset
        triad_system.reset_triad_state(0.750)
        state = triad_system.get_triad_state()
        assert state.crossings == 0
        assert state.z == 0.750
    
    def test_state_is_triad_state_object(self):
        """get_triad_state returns TriadState object"""
        state = triad_system.get_triad_state()
        assert isinstance(state, TriadState)


class TestCrossingDetection:
    """Test threshold crossing detection"""
    
    def setup_method(self):
        triad_system.reset_triad_state(0.800)
    
    def test_crossing_above_high_threshold(self):
        """Crossing above 0.85 is detected"""
        result = triad_system.step(0.86)
        state = triad_system.get_triad_state()
        assert state.crossings == 1
        assert state.above_high == True
    
    def test_no_crossing_below_high_threshold(self):
        """Staying below 0.85 doesn't trigger crossing"""
        triad_system.step(0.84)
        state = triad_system.get_triad_state()
        assert state.crossings == 0
    
    def test_step_returns_dict(self):
        """step() returns a dict with transition info"""
        result = triad_system.step(0.86)
        assert isinstance(result, dict)
        assert 'z' in result
        assert 'crossings' in result


class TestRearmMechanism:
    """Test re-arm mechanism below 0.82"""
    
    def setup_method(self):
        triad_system.reset_triad_state(0.800)
    
    def test_rearm_below_low_threshold(self):
        """Dropping below 0.82 enables next crossing"""
        # First crossing
        triad_system.step(0.86)
        state = triad_system.get_triad_state()
        assert state.crossings == 1
        
        # Re-arm
        triad_system.step(0.81)
        assert state.above_high == False
        
        # Second crossing should work
        triad_system.step(0.86)
        state = triad_system.get_triad_state()
        assert state.crossings == 2
    
    def test_no_rearm_above_low_threshold(self):
        """Staying above 0.82 doesn't re-arm"""
        # First crossing
        triad_system.step(0.86)
        
        # Stay above re-arm threshold
        triad_system.step(0.83)
        
        # Attempt second crossing
        triad_system.step(0.86)
        state = triad_system.get_triad_state()
        # Should still be 1 crossing (no re-arm occurred)
        assert state.crossings == 1


class TestTriadUnlock:
    """Test TRIAD unlock after 3 crossings"""
    
    def setup_method(self):
        triad_system.reset_triad_state(0.800)
    
    def test_three_crossings_unlocks(self):
        """Three crossings unlock TRIAD"""
        # Crossing 1
        triad_system.step(0.86)
        triad_system.step(0.81)  # Re-arm
        
        # Crossing 2
        triad_system.step(0.87)
        triad_system.step(0.81)  # Re-arm
        
        # Crossing 3
        triad_system.step(0.88)
        
        state = triad_system.get_triad_state()
        assert state.crossings >= TRIAD_PASSES_REQUIRED
        assert state.unlocked == True
    
    def test_two_crossings_stays_locked(self):
        """Two crossings keep TRIAD locked"""
        # Crossing 1
        triad_system.step(0.86)
        triad_system.step(0.81)  # Re-arm
        
        # Crossing 2
        triad_system.step(0.87)
        
        state = triad_system.get_triad_state()
        assert state.crossings == 2
        assert state.unlocked == False
    
    def test_unlock_persists(self):
        """Once unlocked, TRIAD stays unlocked"""
        # Full unlock sequence
        triad_system.step(0.86)
        triad_system.step(0.81)
        triad_system.step(0.87)
        triad_system.step(0.81)
        triad_system.step(0.88)
        
        # Move around
        triad_system.step(Z_CRITICAL)
        triad_system.step(0.90)
        
        state = triad_system.get_triad_state()
        assert state.unlocked == True


class TestT6GateControl:
    """Test T6 tier gate lock/unlock based on TRIAD"""
    
    def setup_method(self):
        triad_system.reset_triad_state(0.800)
    
    def test_t6_locked_initially(self):
        """T6 gate is at Z_CRITICAL initially"""
        state = triad_system.get_triad_state()
        assert state.t6_gate == Z_CRITICAL
        assert state.gate_state == T6GateState.CRITICAL
    
    def test_t6_unlocks_with_triad(self):
        """T6 gate changes when TRIAD unlocks"""
        # Unlock TRIAD
        triad_system.step(0.86)
        triad_system.step(0.81)
        triad_system.step(0.87)
        triad_system.step(0.81)
        triad_system.step(0.88)
        
        state = triad_system.get_triad_state()
        assert state.unlocked == True
        assert state.gate_state == T6GateState.TRIAD


class TestBandStateTransitions:
    """Test band state transitions (LOW/MID/HIGH bands)"""
    
    def setup_method(self):
        triad_system.reset_triad_state(0.800)
    
    def test_band_state_available(self):
        """Band state information is available"""
        band = triad_system.get_band_state(0.800)
        assert band is not None
        assert isinstance(band, BandState)
    
    def test_below_low_band(self):
        """z < 0.82 is in BELOW_LOW band"""
        band = triad_system.get_band_state(0.80)
        assert band == BandState.BELOW_LOW
    
    def test_in_band(self):
        """0.82 <= z < 0.85 is in IN_BAND"""
        band = triad_system.get_band_state(0.83)
        assert band == BandState.IN_BAND
    
    def test_above_high_band(self):
        """z >= 0.85 is in ABOVE_HIGH band"""
        band = triad_system.get_band_state(0.87)
        assert band == BandState.ABOVE_HIGH


class TestRunSteps:
    """Test batch step execution"""
    
    def setup_method(self):
        triad_system.reset_triad_state(0.800)
    
    def test_run_steps_sequence(self):
        """run_steps executes sequence correctly"""
        z_sequence = [0.86, 0.81, 0.87, 0.81, 0.88]
        result = triad_system.run_steps(z_sequence)
        
        state = triad_system.get_triad_state()
        assert state.unlocked == True
        assert isinstance(result, dict)
        assert 'steps' in result


class TestTriadHysteresisController:
    """Test the OO wrapper class TriadHysteresisController"""
    
    def test_controller_exists(self):
        """TriadHysteresisController class exists"""
        assert hasattr(triad_system, 'TriadHysteresisController')
    
    def test_controller_can_instantiate(self):
        """Controller can be instantiated"""
        from ucf.core import TriadHysteresisController
        controller = TriadHysteresisController(initial_z=0.800)
        assert controller is not None
    
    def test_controller_properties(self):
        """Controller has expected properties"""
        from ucf.core import TriadHysteresisController
        controller = TriadHysteresisController(initial_z=0.800)
        assert hasattr(controller, 'unlocked')
        assert hasattr(controller, 'crossings')
        assert hasattr(controller, 't6_gate')
    
    def test_controller_unlock_sequence(self):
        """Controller can unlock via step()"""
        from ucf.core import TriadHysteresisController
        controller = TriadHysteresisController(initial_z=0.800)
        
        controller.step(0.86)
        controller.step(0.81)
        controller.step(0.87)
        controller.step(0.81)
        controller.step(0.88)
        
        assert controller.unlocked == True
        assert controller.crossings == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def setup_method(self):
        triad_system.reset_triad_state(0.800)
    
    def test_exact_high_threshold(self):
        """z exactly at 0.85 (TRIAD_HIGH) triggers crossing"""
        triad_system.step(TRIAD_HIGH)
        state = triad_system.get_triad_state()
        assert state.crossings == 1
    
    def test_exact_low_threshold(self):
        """z below 0.82 (TRIAD_LOW) triggers re-arm"""
        triad_system.step(0.86)  # Cross first
        triad_system.step(0.81)  # Below TRIAD_LOW
        state = triad_system.get_triad_state()
        assert state.above_high == False
    
    def test_z_at_critical(self):
        """z at Z_CRITICAL (√3/2 ≈ 0.866) triggers crossing"""
        triad_system.step(Z_CRITICAL)
        state = triad_system.get_triad_state()
        assert state.crossings == 1
    
    def test_history_tracking(self):
        """History of z values is tracked"""
        triad_system.step(0.86)
        triad_system.step(0.81)
        state = triad_system.get_triad_state()
        assert len(state.history) >= 2
