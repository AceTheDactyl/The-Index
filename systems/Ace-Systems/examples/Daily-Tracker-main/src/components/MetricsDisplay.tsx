/**
 * Metrics Display Component
 *
 * Real-time display of DeltaHV metrics with brain region visualization.
 * Shows: Symbolic (S), Resonance (R), Friction (δφ), Stability (H)
 *
 * Each metric displays:
 * - Current value (0-100)
 * - Associated brain regions with activation glyphs
 * - Real-time updates from metricsHub
 */

import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp } from 'lucide-react';
import { metricsHub, type EnhancedDeltaHVState } from '../lib/metricsHub';

interface MetricsDisplayProps {
  compact?: boolean;
  showBrainRegions?: boolean;
  onMetricClick?: (metric: 'symbolic' | 'resonance' | 'friction' | 'stability') => void;
}

export const MetricsDisplay: React.FC<MetricsDisplayProps> = ({
  compact = false,
  showBrainRegions = true,
  onMetricClick,
}) => {
  const [state, setState] = useState<EnhancedDeltaHVState | null>(null);
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);

  useEffect(() => {
    // Subscribe to real-time updates
    const unsubscribe = metricsHub.subscribe(setState);
    return unsubscribe;
  }, []);

  if (!state) {
    return (
      <div className="p-4 bg-gray-950/60 rounded-xl border border-gray-800">
        <div className="flex items-center gap-2 text-gray-500">
          <Activity className="w-4 h-4 animate-pulse" />
          <span className="text-sm">Initializing metrics...</span>
        </div>
      </div>
    );
  }

  const metrics = [
    {
      id: 'symbolic' as const,
      label: 'Symbolic (S)',
      value: state.symbolicDensity,
      color: 'purple',
      icon: '✨',
      description: 'Meaning, intention, symbolic density',
      regions: state.brainActivation.symbolic,
    },
    {
      id: 'resonance' as const,
      label: 'Resonance (R)',
      value: state.resonanceCoupling,
      color: 'cyan',
      icon: '🎯',
      description: 'Alignment between intention and action',
      regions: state.brainActivation.resonance,
    },
    {
      id: 'friction' as const,
      label: 'Friction (δφ)',
      value: state.frictionCoefficient,
      color: 'orange',
      icon: '🌧️',
      description: 'Resistance, missed tasks, delays',
      regions: state.brainActivation.friction,
      inverted: true, // Lower is better
    },
    {
      id: 'stability' as const,
      label: 'Stability (H)',
      value: state.harmonicStability,
      color: 'green',
      icon: '⚖️',
      description: 'Consistency and rhythmic stability',
      regions: state.brainActivation.stability,
    },
  ];

  const getColorClasses = (color: string, value: number, inverted = false) => {
    const effectiveValue = inverted ? 100 - value : value;
    const intensity = effectiveValue > 60 ? 'high' : effectiveValue > 30 ? 'medium' : 'low';

    const colors: Record<string, Record<string, { bg: string; text: string; border: string }>> = {
      purple: {
        high: { bg: 'bg-purple-500/30', text: 'text-purple-300', border: 'border-purple-500/50' },
        medium: { bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/30' },
        low: { bg: 'bg-purple-500/10', text: 'text-purple-500', border: 'border-purple-500/20' },
      },
      cyan: {
        high: { bg: 'bg-cyan-500/30', text: 'text-cyan-300', border: 'border-cyan-500/50' },
        medium: { bg: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500/30' },
        low: { bg: 'bg-cyan-500/10', text: 'text-cyan-500', border: 'border-cyan-500/20' },
      },
      orange: {
        high: { bg: 'bg-orange-500/30', text: 'text-orange-300', border: 'border-orange-500/50' },
        medium: { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/30' },
        low: { bg: 'bg-orange-500/10', text: 'text-orange-500', border: 'border-orange-500/20' },
      },
      green: {
        high: { bg: 'bg-green-500/30', text: 'text-green-300', border: 'border-green-500/50' },
        medium: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30' },
        low: { bg: 'bg-green-500/10', text: 'text-green-500', border: 'border-green-500/20' },
      },
    };

    return colors[color][intensity];
  };

  if (compact) {
    return (
      <div className="flex items-center gap-4 p-3 bg-gray-950/60 rounded-xl border border-gray-800">
        {metrics.map((metric) => (
          <button
            key={metric.id}
            onClick={() => onMetricClick?.(metric.id)}
            className="flex items-center gap-2 group"
          >
            <span className="text-lg">{metric.icon}</span>
            <div className="text-right">
              <div className={`text-lg font-light ${getColorClasses(metric.color, metric.value, metric.inverted).text}`}>
                {metric.value}%
              </div>
            </div>
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2 text-gray-500">
          {state.isLive && (
            <span className="flex items-center gap-1 text-xs">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              Live
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-medium text-white">DeltaHV Metrics</span>
        </div>
        <div className="flex items-center gap-3">
          <span className={`text-xs px-2 py-1 rounded-full ${
            state.fieldState === 'coherent' ? 'bg-green-500/20 text-green-300' :
            state.fieldState === 'transitioning' ? 'bg-yellow-500/20 text-yellow-300' :
            state.fieldState === 'fragmented' ? 'bg-orange-500/20 text-orange-300' :
            'bg-gray-500/20 text-gray-300'
          }`}>
            {state.fieldState}
          </span>
          {state.isLive && (
            <span className="flex items-center gap-1 text-xs text-gray-500">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              {state.updateCount} updates
            </span>
          )}
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="p-4 grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map((metric) => {
          const colorClasses = getColorClasses(metric.color, metric.value, metric.inverted);
          const isExpanded = expandedMetric === metric.id;

          return (
            <button
              key={metric.id}
              onClick={() => {
                setExpandedMetric(isExpanded ? null : metric.id);
                onMetricClick?.(metric.id);
              }}
              className={`p-3 rounded-lg border transition-all text-left ${colorClasses.bg} ${colorClasses.border} hover:scale-[1.02]`}
            >
              {/* Metric Header */}
              <div className="flex items-center justify-between mb-2">
                <span className="text-lg">{metric.icon}</span>
                <span className={`text-xl font-light ${colorClasses.text}`}>
                  {metric.value}%
                </span>
              </div>

              {/* Label */}
              <div className="text-sm text-gray-400 mb-1">{metric.label}</div>

              {/* Progress Bar */}
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    metric.color === 'purple' ? 'bg-purple-500' :
                    metric.color === 'cyan' ? 'bg-cyan-500' :
                    metric.color === 'orange' ? 'bg-orange-500' :
                    'bg-green-500'
                  }`}
                  style={{ width: `${metric.value}%` }}
                />
              </div>

              {/* Brain Regions (if enabled and expanded or always visible) */}
              {showBrainRegions && (
                <div className={`mt-3 space-y-1 ${isExpanded ? '' : 'hidden md:block'}`}>
                  {metric.regions.slice(0, isExpanded ? undefined : 3).map((region, i) => (
                    <div
                      key={i}
                      className="text-xs text-gray-500 flex items-center gap-1"
                    >
                      <span className="opacity-70">{region}</span>
                    </div>
                  ))}
                  {!isExpanded && metric.regions.length > 3 && (
                    <div className="text-xs text-gray-600">
                      +{metric.regions.length - 3} more
                    </div>
                  )}
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Music Influence Footer */}
      {state.musicInfluence && state.musicInfluence.authorshipScore > 0 && (
        <div className="px-4 py-3 border-t border-gray-800 bg-gray-900/50">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-4">
              <span className="text-gray-400">
                🎵 Music Authorship: <span className="text-cyan-400">{state.musicInfluence.authorshipScore}%</span>
              </span>
              <span className="text-gray-400">
                Skip Ratio: <span className="text-purple-400">{Math.round(state.musicInfluence.skipRatio * 100)}%</span>
              </span>
            </div>
            <span className={`text-xs flex items-center gap-1 ${
              state.musicInfluence.emotionalTrajectory === 'rising' ? 'text-green-400' :
              state.musicInfluence.emotionalTrajectory === 'processing' ? 'text-yellow-400' :
              'text-gray-400'
            }`}>
              {state.musicInfluence.emotionalTrajectory === 'rising' && <TrendingUp className="w-3 h-3" />}
              {state.musicInfluence.emotionalTrajectory === 'processing' && <Activity className="w-3 h-3" />}
              {state.musicInfluence.emotionalTrajectory}
            </span>
          </div>
        </div>
      )}

      {/* Last Action */}
      <div className="px-4 py-2 border-t border-gray-800 text-xs text-gray-600">
        Last: {state.lastAction}
      </div>
    </div>
  );
};

/**
 * Compact inline metrics for headers/nav
 */
export const InlineMetrics: React.FC<{ className?: string }> = ({ className = '' }) => {
  const [state, setState] = useState<EnhancedDeltaHVState | null>(null);

  useEffect(() => {
    const unsubscribe = metricsHub.subscribe(setState);
    return unsubscribe;
  }, []);

  if (!state) return null;

  return (
    <div className={`flex items-center gap-3 text-sm ${className}`}>
      <span className="flex items-center gap-1">
        <span>✨</span>
        <span className="text-purple-400">{state.symbolicDensity}</span>
      </span>
      <span className="flex items-center gap-1">
        <span>🎯</span>
        <span className="text-cyan-400">{state.resonanceCoupling}</span>
      </span>
      <span className="flex items-center gap-1">
        <span>🌧️</span>
        <span className="text-orange-400">{state.frictionCoefficient}</span>
      </span>
      <span className="flex items-center gap-1">
        <span>⚖️</span>
        <span className="text-green-400">{state.harmonicStability}</span>
      </span>
    </div>
  );
};

export default MetricsDisplay;
