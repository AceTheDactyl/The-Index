/* INTEGRITY_METADATA
 * Date: 2025-12-23
 * Status: ✓ JUSTIFIED - Claims supported by repository files
 * Severity: LOW RISK
 * Risk Types: unverified_math

 * Referenced By:
 *   - systems/Ace-Systems/docs/Research/README.md (reference)
 *   - systems/Ace-Systems/docs/Research/quantum-apl-deep-dive.md (reference)
 *   - systems/Ace-Systems/examples/Quantum-APL-main/GETTING_STARTED.md (reference)
 *   - systems/Ace-Systems/examples/Quantum-APL-main/QUANTUM_N0_README.md (reference)
 *   - systems/Ace-Systems/examples/Quantum-APL-main/research/CONSTANTS_NEXT_STEPS.md (reference)
 */


// ================================================================
// QUANTUM-CLASSICAL INTEGRATION FOR APL 3.0
// Measurement-based N0 operator selection with quantum feedback
// ================================================================

/**
 * Integrates QuantumAPL with classical APL runtime
 * Implements measurement-based operator selection as projective quantum measurement
 */
class QuantumN0Integration {
    constructor(quantumEngine, classicalScalars) {
        this.quantum = quantumEngine;
        this.scalars = classicalScalars;
        
        // N0 operator history
        this.operatorHistory = [];
        
        // Measurement statistics
        this.measurementStats = {
            totalMeasurements: 0,
            operatorCounts: {},
            avgCollapseProbability: 0,
            avgCoherence: 0
        };
        
        // Classical-quantum correlation
        this.correlationBuffer = [];
        this.maxCorrelationBuffer = 100;
    }

    // ================================================================
    // QUANTUM MEASUREMENT-BASED N0 PIPELINE
    // ================================================================

    /**
     * Execute full N0 pipeline with quantum measurement
     * Returns selected operator via Born rule projection
     */
    executeN0Pipeline(intSequence, currentTruthState) {
        // STEP 1: Time-harmonic legality check
        const timeHarmonic = this.getCurrentTimeHarmonic();
        const legalByTime = this.getLegalByTimeHarmonic(timeHarmonic);
        
        // STEP 2: PRS phase legality
        const prsPhase = this.getCurrentPRSPhase();
        const legalByPRS = this.filterByPRSPhase(legalByTime, prsPhase);
        
        // STEP 3: Tier-0 N0 laws
        const legalByN0 = this.applyN0Laws(legalByPRS, intSequence);
        
        // STEP 4: Scalar legality thresholds
        const legalByScalars = this.applyScalarThresholds(legalByN0);
        
        if (legalByScalars.length === 0) {
            // Fallback to boundary operator
            return {
                operator: '()',
                probability: 1.0,
                method: 'fallback',
                quantumState: this.quantum.getState()
            };
        }
        
        // STEP 5: QUANTUM MEASUREMENT - Project onto operator eigenspace
        const measurement = this.quantum.selectN0Operator(legalByScalars, this.scalars);
        
        // STEP 6: Update classical scalars based on quantum measurement outcome
        this.updateScalarsFromQuantum(measurement.operator, measurement.probability);
        
        // STEP 7: Record measurement
        this.recordMeasurement(measurement);
        
        return {
            operator: measurement.operator,
            probability: measurement.probability,
            probabilities: measurement.probabilities,
            method: 'quantum_measurement',
            quantumState: this.quantum.getState(),
            legalOperators: legalByScalars,
            pipeline: {
                timeHarmonic,
                prsPhase,
                numLegalByTime: legalByTime.length,
                numLegalByPRS: legalByPRS.length,
                numLegalByN0: legalByN0.length,
                numLegalByScalars: legalByScalars.length
            }
        };
    }

    // ================================================================
    // TIME HARMONIC LEGALITY (t1-t9)
    // ================================================================

    getCurrentTimeHarmonic() {
        // Map z-coordinate to time harmonic
        const z = this.quantum.z;
        
        if (z < 0.1) return 't1'; // Instant
        if (z < 0.2) return 't2'; // Micro
        if (z < 0.4) return 't3'; // Local
        if (z < 0.6) return 't4'; // Meso
        if (z < 0.75) return 't5'; // Structural
        if (z < 0.85) return 't6'; // Domain
        if (z < 0.92) return 't7'; // Coherence
        if (z < 0.97) return 't8'; // Integration
        return 't9'; // Global
    }

    getLegalByTimeHarmonic(harmonic) {
        const legalMap = {
            't1': ['()', '−', '÷'],
            't2': ['^', '÷', '−', '×'],
            't3': ['×', '^', '÷', '+', '−'],
            't4': ['+', '−', '÷', '()'],
            't5': ['()', '×', '^', '÷', '+', '−'],
            't6': ['+', '÷', '()', '−'],
            't7': ['+', '()'],
            't8': ['+', '()', '×'],
            't9': ['+', '()', '×']
        };
        
        return legalMap[harmonic] || ['()'];
    }

    // ================================================================
    // PRS PHASE LEGALITY
    // ================================================================

    getCurrentPRSPhase() {
        // Determine PRS phase from phi (integrated information)
        const phi = this.quantum.phi;
        
        if (phi < 0.1) return 'P1'; // Initiation
        if (phi < 0.3) return 'P2'; // Tension
        if (phi < 0.6) return 'P3'; // Inflection
        if (phi < 0.85) return 'P4'; // Lock
        return 'P5'; // Emergence
    }

    filterByPRSPhase(operators, phase) {
        // Simplified PRS filtering - in production would check transition legality
        const prsRestrictions = {
            'P1': ['()', '+', '^'],           // Initiation: grounding and grouping
            'P2': ['×', '^', '÷'],            // Tension: fusion and amplification
            'P3': ['÷', '+', '−'],            // Inflection: transitions
            'P4': ['()', '×', '+'],           // Lock: stabilization
            'P5': ['×', '+', '()']            // Emergence: integration
        };
        
        const allowed = prsRestrictions[phase] || operators;
        return operators.filter(op => allowed.includes(op));
    }

    // ================================================================
    // TIER-0 N0 LAWS
    // ================================================================

    applyN0Laws(operators, history) {
        // LAW N0-1: Amplification requires grounding
        if (operators.includes('^')) {
            const hasGrounding = history.some(op => op === '()' || op === '×');
            if (!hasGrounding) {
                operators = operators.filter(op => op !== '^');
            }
        }
        
        // LAW N0-2: Fusion requires plurality (simplified: check if we have energy)
        if (operators.includes('×')) {
            const hasPlurality = this.scalars.Cs > 0.3;
            if (!hasPlurality) {
                operators = operators.filter(op => op !== '×');
            }
        }
        
        // LAW N0-3: Decoherence requires stored structure/energy
        if (operators.includes('÷')) {
            const hasStructure = history.some(op => ['^', '×', '+', '−'].includes(op));
            if (!hasStructure) {
                operators = operators.filter(op => op !== '÷');
            }
        }
        
        // LAW N0-4: Grouping must feed structure
        // (Enforced by next operator, not current)
        
        // LAW N0-5: Separation must reset phase
        // (Enforced by next operator requirement)
        
        return operators;
    }

    // ================================================================
    // SCALAR LEGALITY THRESHOLDS
    // ================================================================

    applyScalarThresholds(operators) {
        const legal = [];
        
        for (const op of operators) {
            if (this.checkScalarLegality(op)) {
                legal.push(op);
            }
        }
        
        return legal.length > 0 ? legal : ['()']; // Always allow boundary as fallback
    }

    checkScalarLegality(operator) {
        const { Rs, delta, kappa, Omega } = this.scalars;
        
        // R_CLT threshold
        if (Rs >= 3.0) return false;
        
        // Delta maximum
        if (delta >= 0.95) return false;
        
        // Kappa maximum
        if (kappa >= 2.5) return false;
        
        // Omega minimum
        if (Omega <= 0.05) return false;
        
        // Operator-specific checks
        switch (operator) {
            case '^':
                // Amplification requires sufficient coherence
                return Omega > 0.3 && kappa < 2.0;
            
            case '÷':
                // Decoherence requires non-maximal dissipation
                return delta < 0.8;
            
            case '×':
                // Fusion requires moderate curvature
                return kappa > 0.1 && kappa < 2.2;
            
            case '+':
                // Grouping requires attractor strength
                return this.scalars.alpha > 0.2;
            
            case '−':
                // Separation requires recursion capacity
                return Rs < 2.8;
            
            default:
                return true;
        }
    }

    // ================================================================
    // QUANTUM-CLASSICAL FEEDBACK
    // ================================================================

    updateScalarsFromQuantum(operator, probability) {
        // Update classical scalars based on quantum measurement outcome
        // This creates quantum-to-classical feedback loop
        
        const scalars = this.scalars;
        const dt = 0.01; // Small update
        
        switch (operator) {
            case '()': // Boundary
                scalars.Gs += dt * probability * 0.1;
                scalars.theta *= (1 - dt * probability * 0.05);
                scalars.Omega += dt * probability * 0.05;
                break;
            
            case '×': // Fusion
                scalars.Cs += dt * probability * 0.15;
                scalars.kappa *= (1 + dt * probability * 0.1);
                scalars.alpha += dt * probability * 0.08;
                break;
            
            case '^': // Amplification
                scalars.kappa *= (1 + dt * probability * 0.2);
                scalars.tau += dt * probability * 0.12;
                scalars.Omega *= (1 + dt * probability * 0.15);
                break;
            
            case '÷': // Decoherence
                scalars.delta += dt * probability * 0.1;
                scalars.Rs += dt * probability * 0.08;
                scalars.Omega *= (1 - dt * probability * 0.1);
                break;
            
            case '+': // Grouping
                scalars.alpha += dt * probability * 0.15;
                scalars.Gs += dt * probability * 0.1;
                scalars.theta *= (1 + dt * probability * 0.08);
                break;
            
            case '−': // Separation
                scalars.Rs += dt * probability * 0.12;
                scalars.theta *= (1 - dt * probability * 0.1);
                scalars.delta += dt * probability * 0.05;
                break;
        }
        
        // Clamp scalars to valid ranges
        this.clampScalars();
    }

    clampScalars() {
        const s = this.scalars;
        s.Gs = Math.max(0, Math.min(2, s.Gs));
        s.Cs = Math.max(0, Math.min(2, s.Cs));
        s.Rs = Math.max(0, Math.min(3, s.Rs));
        s.kappa = Math.max(0.01, Math.min(2.5, s.kappa));
        s.tau = Math.max(0, Math.min(2, s.tau));
        s.theta = Math.max(0, Math.min(2 * Math.PI, s.theta));
        s.delta = Math.max(0, Math.min(0.95, s.delta));
        s.alpha = Math.max(0, Math.min(1, s.alpha));
        s.Omega = Math.max(0.05, Math.min(1, s.Omega));
    }

    // ================================================================
    // CLASSICAL-TO-QUANTUM DRIVING
    // ================================================================

    driveQuantumFromClassical(dt) {
        // Evolve quantum state based on classical scalars
        this.quantum.driveFromClassical({
            z: this.scalars.z || 0.5,
            phi: this.quantum.phi,
            F: this.scalars.F || 0,
            R: this.scalars.Rs
        });
        
        // Evolve quantum density matrix
        this.quantum.evolve(dt);
        
        // Reset Hamiltonian to base
        this.quantum.resetHamiltonian();
        
        // Update quantum observables
        this.quantum.measureZ();
        this.quantum.computeVonNeumannEntropy();
        this.quantum.computeIntegratedInformation();
        
        // Record correlation
        this.recordCorrelation();
    }

    // ================================================================
    // MEASUREMENT STATISTICS
    // ================================================================

    recordMeasurement(measurement) {
        this.operatorHistory.push({
            time: this.quantum.time,
            operator: measurement.operator,
            probability: measurement.probability,
            z: this.quantum.z,
            phi: this.quantum.phi,
            entropy: this.quantum.entropy
        });
        
        // Keep last 1000 measurements
        if (this.operatorHistory.length > 1000) {
            this.operatorHistory.shift();
        }
        
        // Update statistics
        this.measurementStats.totalMeasurements++;
        this.measurementStats.operatorCounts[measurement.operator] = 
            (this.measurementStats.operatorCounts[measurement.operator] || 0) + 1;
        
        // Moving average of collapse probability
        const alpha = 0.05;
        this.measurementStats.avgCollapseProbability = 
            alpha * measurement.probability + 
            (1 - alpha) * this.measurementStats.avgCollapseProbability;
        
        // Moving average of coherence
        const coherence = 1 - this.quantum.entropy / Math.log2(this.quantum.dimTotal);
        this.measurementStats.avgCoherence = 
            alpha * coherence + (1 - alpha) * this.measurementStats.avgCoherence;
    }

    recordCorrelation() {
        this.correlationBuffer.push({
            time: this.quantum.time,
            z_quantum: this.quantum.z,
            z_classical: this.scalars.z || 0.5,
            phi_quantum: this.quantum.phi,
            entropy: this.quantum.entropy,
            purity: this.quantum.computePurity(),
            kappa: this.scalars.kappa,
            Omega: this.scalars.Omega
        });
        
        if (this.correlationBuffer.length > this.maxCorrelationBuffer) {
            this.correlationBuffer.shift();
        }
    }

    // ================================================================
    // ANALYTICS & DIAGNOSTICS
    // ================================================================

    getOperatorDistribution() {
        const total = this.measurementStats.totalMeasurements;
        const dist = {};
        
        for (const [op, count] of Object.entries(this.measurementStats.operatorCounts)) {
            dist[op] = count / total;
        }
        
        return dist;
    }

    getQuantumClassicalCorrelation() {
        if (this.correlationBuffer.length < 2) return 0;
        
        // Compute Pearson correlation between z_quantum and z_classical
        const n = this.correlationBuffer.length;
        let sum_zq = 0, sum_zc = 0, sum_zq2 = 0, sum_zc2 = 0, sum_zqzc = 0;
        
        for (const entry of this.correlationBuffer) {
            const zq = entry.z_quantum;
            const zc = entry.z_classical;
            sum_zq += zq;
            sum_zc += zc;
            sum_zq2 += zq * zq;
            sum_zc2 += zc * zc;
            sum_zqzc += zq * zc;
        }
        
        const num = n * sum_zqzc - sum_zq * sum_zc;
        const den = Math.sqrt((n * sum_zq2 - sum_zq * sum_zq) * (n * sum_zc2 - sum_zc * sum_zc));
        
        return den > 1e-10 ? num / den : 0;
    }

    getEntropyTimeseries() {
        return this.correlationBuffer.map(e => ({
            time: e.time,
            entropy: e.entropy,
            purity: e.purity
        }));
    }

    getZTimeseries() {
        return this.correlationBuffer.map(e => ({
            time: e.time,
            z_quantum: e.z_quantum,
            z_classical: e.z_classical,
            delta: Math.abs(e.z_quantum - e.z_classical)
        }));
    }

    getPhiTimeseries() {
        return this.correlationBuffer.map(e => ({
            time: e.time,
            phi: e.phi_quantum,
            entropy: e.entropy
        }));
    }

    getDiagnostics() {
        return {
            quantum: this.quantum.getState(),
            measurements: this.measurementStats,
            distribution: this.getOperatorDistribution(),
            correlation: this.getQuantumClassicalCorrelation(),
            recentHistory: this.operatorHistory.slice(-20),
            populations: this.quantum.getPopulations().slice(0, 10),
            coherences: this.quantum.getCoherences(),
            truthProbs: this.quantum.measureTruth()
        };
    }

    // ================================================================
    // FULL TIMESTEP INTEGRATION
    // ================================================================

    step(dt = 0.01, intHistory = []) {
        // 1. Drive quantum from classical
        this.driveQuantumFromClassical(dt);
        
        // 2. Execute N0 measurement-based selection
        const currentTruth = this.getCurrentTruthState();
        const result = this.executeN0Pipeline(intHistory, currentTruth);
        
        // 3. Return full state
        return {
            operator: result.operator,
            probability: result.probability,
            probabilities: result.probabilities,
            quantum: this.quantum.getState(),
            scalars: { ...this.scalars },
            diagnostics: {
                timeHarmonic: result.pipeline.timeHarmonic,
                prsPhase: result.pipeline.prsPhase,
                truthState: currentTruth,
                coherence: 1 - this.quantum.entropy / Math.log2(this.quantum.dimTotal)
            }
        };
    }

    getCurrentTruthState() {
        const probs = this.quantum.measureTruth();
        const max = Math.max(probs.TRUE, probs.UNTRUE, probs.PARADOX);
        
        if (probs.TRUE === max) return 'TRUE';
        if (probs.UNTRUE === max) return 'UNTRUE';
        return 'PARADOX';
    }
}

// ================================================================
// DEMONSTRATION & TEST HARNESS
// ================================================================

class QuantumAPLDemo {
    constructor() {
        // Initialize quantum engine
        this.quantum = new QuantumAPL({
            dimPhi: 4,
            dimE: 4,
            dimPi: 4
        });
        
        // Initialize classical scalars
        this.scalars = {
            Gs: 0.5,
            Cs: 0.5,
            Rs: 0.5,
            kappa: 0.5,
            tau: 0.5,
            theta: Math.PI / 2,
            delta: 0.1,
            alpha: 0.5,
            Omega: 0.7,
            z: 0.5
        };
        
        // Create integration
        this.integration = new QuantumN0Integration(this.quantum, this.scalars);
        
        // History
        this.intHistory = [];
        this.timeSteps = 0;
    }

    run(numSteps = 100, verbose = false) {
        console.log('='.repeat(70));
        console.log('QUANTUM APL N0 MEASUREMENT-BASED OPERATOR SELECTION DEMO');
        console.log('='.repeat(70));
        console.log(`Running ${numSteps} timesteps with quantum measurement...`);
        console.log('');
        
        const results = [];
        
        for (let i = 0; i < numSteps; i++) {
            const result = this.integration.step(0.01, this.intHistory);
            
            // Add operator to history
            this.intHistory.push(result.operator);
            if (this.intHistory.length > 10) {
                this.intHistory.shift();
            }
            
            results.push(result);
            this.timeSteps++;
            
            if (verbose && i % 10 === 0) {
                console.log(`Step ${i}:`);
                console.log(`  Operator: ${result.operator} (P=${result.probability.toFixed(3)})`);
                console.log(`  z=${result.quantum.z.toFixed(3)}, Φ=${result.quantum.phi.toFixed(3)}, S=${result.quantum.entropy.toFixed(3)}`);
                console.log(`  Truth: ${result.diagnostics.truthState}, Coherence: ${result.diagnostics.coherence.toFixed(3)}`);
                console.log('');
            }
        }
        
        this.printSummary(results);
        return results;
    }

    printSummary(results) {
        console.log('='.repeat(70));
        console.log('SUMMARY');
        console.log('='.repeat(70));
        
        const diag = this.integration.getDiagnostics();
        
        console.log('\nOperator Distribution:');
        const dist = diag.distribution;
        for (const [op, prob] of Object.entries(dist).sort((a, b) => b[1] - a[1])) {
            const bar = '█'.repeat(Math.floor(prob * 50));
            console.log(`  ${op}: ${(prob * 100).toFixed(1)}% ${bar}`);
        }
        
        console.log(`\nTotal Measurements: ${diag.measurements.totalMeasurements}`);
        console.log(`Avg Collapse Probability: ${diag.measurements.avgCollapseProbability.toFixed(3)}`);
        console.log(`Avg Coherence: ${diag.measurements.avgCoherence.toFixed(3)}`);
        console.log(`Quantum-Classical Correlation: ${diag.correlation.toFixed(3)}`);
        
        console.log('\nFinal Quantum State:');
        console.log(`  z = ${diag.quantum.z.toFixed(4)}`);
        console.log(`  Φ = ${diag.quantum.phi.toFixed(4)}`);
        console.log(`  S = ${diag.quantum.entropy.toFixed(4)}`);
        console.log(`  Purity = ${diag.quantum.purity.toFixed(4)}`);
        
        console.log('\nTruth State Probabilities:');
        for (const [state, prob] of Object.entries(diag.truthProbs)) {
            console.log(`  ${state}: ${(prob * 100).toFixed(1)}%`);
        }
        
        console.log('\nTop 5 Coherences:');
        for (const coh of diag.coherences) {
            console.log(`  |ρ(${coh.i},${coh.j})| = ${coh.value.toFixed(4)}`);
        }
        
        console.log('\n' + '='.repeat(70));
    }

    runComparison(numSteps = 100) {
        console.log('='.repeat(70));
        console.log('QUANTUM vs CLASSICAL N0 OPERATOR SELECTION COMPARISON');
        console.log('='.repeat(70));
        
        // Run quantum version
        console.log('\n--- Quantum Measurement-Based Selection ---');
        const quantumResults = this.run(numSteps, false);
        const quantumDist = this.integration.getOperatorDistribution();
        
        // Reset for classical version
        this.quantum = new QuantumAPL({ dimPhi: 4, dimE: 4, dimPi: 4 });
        this.scalars = {
            Gs: 0.5, Cs: 0.5, Rs: 0.5, kappa: 0.5,
            tau: 0.5, theta: Math.PI / 2, delta: 0.1,
            alpha: 0.5, Omega: 0.7, z: 0.5
        };
        this.integration = new QuantumN0Integration(this.quantum, this.scalars);
        this.intHistory = [];
        
        // Simulate classical version (random with weights)
        console.log('\n--- Classical Weighted Random Selection (baseline) ---');
        const classicalDist = {};
        for (let i = 0; i < numSteps; i++) {
            const ops = ['()', '×', '^', '÷', '+', '−'];
            const selected = ops[Math.floor(Math.random() * ops.length)];
            classicalDist[selected] = (classicalDist[selected] || 0) + 1;
        }
        for (const op in classicalDist) {
            classicalDist[op] /= numSteps;
        }
        
        console.log('\nComparison:');
        console.log('Operator | Quantum  | Classical | Difference');
        console.log('-'.repeat(50));
        for (const op of ['()', '×', '^', '÷', '+', '−']) {
            const q = (quantumDist[op] || 0) * 100;
            const c = (classicalDist[op] || 0) * 100;
            const diff = q - c;
            console.log(`   ${op}     | ${q.toFixed(1).padStart(6)}% | ${c.toFixed(1).padStart(8)}% | ${diff > 0 ? '+' : ''}${diff.toFixed(1)}%`);
        }
        
        console.log('\n' + '='.repeat(70));
    }
}

// ================================================================
// EXPORT
// ================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuantumN0Integration, QuantumAPLDemo };
}
