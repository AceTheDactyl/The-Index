/**
 * Analytics Dashboard
 *
 * Visual charts and insights for rhythm patterns over time:
 * - Daily completion rates (line chart)
 * - Wave distribution (pie chart)
 * - ΔHV trends (area chart)
 * - Category breakdown (bar chart)
 * - Rhythm state distribution (donut chart)
 */

import React, { useState, useMemo } from 'react';
import {
  X, TrendingUp, PieChart, BarChart3, Activity, Calendar, Zap
} from 'lucide-react';

// Types
interface CheckIn {
  id: string;
  category: string;
  task: string;
  waveId?: string;
  slot: string;
  loggedAt: string;
  done: boolean;
  isAnchor?: boolean;
}

interface Wave {
  id: string;
  name: string;
  color: string;
}

interface AnalyticsDashboardProps {
  checkIns: CheckIn[];
  waves: Wave[];
  onClose: () => void;
}

// Color mapping
const COLORS = {
  cyan: '#22d3ee',
  purple: '#a855f7',
  blue: '#3b82f6',
  orange: '#f97316',
  pink: '#ec4899',
  green: '#22c55e',
  amber: '#f59e0b',
  rose: '#f43f5e',
  gray: '#6b7280'
};

const CATEGORY_COLORS: Record<string, string> = {
  Workout: COLORS.pink,
  Moderation: COLORS.cyan,
  Meditation: COLORS.green,
  Emotion: COLORS.purple,
  General: COLORS.gray,
  Journal: COLORS.blue,
  Anchor: COLORS.amber
};

// Utility functions
const sameDay = (a: Date, b: Date): boolean =>
  a.getFullYear() === b.getFullYear() &&
  a.getMonth() === b.getMonth() &&
  a.getDate() === b.getDate();

const formatDate = (date: Date): string => {
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

const getDateRange = (days: number): Date[] => {
  const dates: Date[] = [];
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    dates.push(date);
  }

  return dates;
};

/**
 * Simple Line Chart Component
 */
function LineChart({
  data,
  width = 300,
  height = 150,
  color = COLORS.cyan,
  showArea = false
}: {
  data: { label: string; value: number }[];
  width?: number;
  height?: number;
  color?: string;
  showArea?: boolean;
}) {
  if (data.length === 0) return null;

  const padding = { top: 10, right: 10, bottom: 25, left: 35 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const maxValue = Math.max(...data.map(d => d.value), 1);
  const minValue = Math.min(...data.map(d => d.value), 0);
  const range = maxValue - minValue || 1;

  const points = data.map((d, i) => ({
    x: padding.left + (i / (data.length - 1 || 1)) * chartWidth,
    y: padding.top + chartHeight - ((d.value - minValue) / range) * chartHeight
  }));

  const linePath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
    .join(' ');

  const areaPath = linePath +
    ` L ${points[points.length - 1].x} ${padding.top + chartHeight}` +
    ` L ${points[0].x} ${padding.top + chartHeight} Z`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map(ratio => (
        <line
          key={ratio}
          x1={padding.left}
          y1={padding.top + chartHeight * (1 - ratio)}
          x2={padding.left + chartWidth}
          y2={padding.top + chartHeight * (1 - ratio)}
          stroke="#374151"
          strokeWidth="1"
          strokeDasharray="4 4"
        />
      ))}

      {/* Y-axis labels */}
      {[0, 0.5, 1].map(ratio => (
        <text
          key={ratio}
          x={padding.left - 5}
          y={padding.top + chartHeight * (1 - ratio) + 4}
          textAnchor="end"
          fontSize="10"
          fill="#6b7280"
        >
          {Math.round(minValue + range * ratio)}
        </text>
      ))}

      {/* Area fill */}
      {showArea && (
        <path
          d={areaPath}
          fill={color}
          fillOpacity="0.1"
        />
      )}

      {/* Line */}
      <path
        d={linePath}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Data points */}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r="3"
          fill={color}
        />
      ))}

      {/* X-axis labels (show every other for space) */}
      {data.map((d, i) => (
        i % Math.ceil(data.length / 5) === 0 && (
          <text
            key={i}
            x={points[i].x}
            y={height - 5}
            textAnchor="middle"
            fontSize="9"
            fill="#6b7280"
          >
            {d.label}
          </text>
        )
      ))}
    </svg>
  );
}

/**
 * Donut Chart Component
 */
function DonutChart({
  data,
  size = 120,
  thickness = 20
}: {
  data: { label: string; value: number; color: string }[];
  size?: number;
  thickness?: number;
}) {
  const total = data.reduce((sum, d) => sum + d.value, 0);
  if (total === 0) {
    return (
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={(size - thickness) / 2}
          fill="none"
          stroke="#374151"
          strokeWidth={thickness}
        />
      </svg>
    );
  }

  const radius = (size - thickness) / 2;
  const circumference = 2 * Math.PI * radius;

  let currentOffset = 0;

  return (
    <svg width={size} height={size} className="transform -rotate-90">
      {data.map((d, i) => {
        const percentage = d.value / total;
        const strokeLength = percentage * circumference;
        const offset = currentOffset;
        currentOffset += strokeLength;

        return (
          <circle
            key={i}
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={d.color}
            strokeWidth={thickness}
            strokeDasharray={`${strokeLength} ${circumference - strokeLength}`}
            strokeDashoffset={-offset}
            strokeLinecap="round"
          />
        );
      })}
    </svg>
  );
}

/**
 * Bar Chart Component
 */
function BarChart({
  data,
  width = 300,
  height = 150,
  horizontal = false
}: {
  data: { label: string; value: number; color: string }[];
  width?: number;
  height?: number;
  horizontal?: boolean;
}) {
  if (data.length === 0) return null;

  const maxValue = Math.max(...data.map(d => d.value), 1);
  const padding = horizontal
    ? { top: 5, right: 10, bottom: 5, left: 80 }
    : { top: 10, right: 10, bottom: 40, left: 30 };

  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  if (horizontal) {
    const barHeight = Math.min(25, chartHeight / data.length - 5);

    return (
      <svg width={width} height={height}>
        {data.map((d, i) => {
          const barWidth = (d.value / maxValue) * chartWidth;
          const y = padding.top + (i * (chartHeight / data.length));

          return (
            <g key={i}>
              <text
                x={padding.left - 5}
                y={y + barHeight / 2 + 4}
                textAnchor="end"
                fontSize="11"
                fill="#9ca3af"
              >
                {d.label}
              </text>
              <rect
                x={padding.left}
                y={y}
                width={barWidth}
                height={barHeight}
                rx="3"
                fill={d.color}
                fillOpacity="0.8"
              />
              <text
                x={padding.left + barWidth + 5}
                y={y + barHeight / 2 + 4}
                fontSize="11"
                fill="#d1d5db"
              >
                {d.value}
              </text>
            </g>
          );
        })}
      </svg>
    );
  }

  const barWidth = Math.min(40, chartWidth / data.length - 10);

  return (
    <svg width={width} height={height}>
      {/* Y-axis lines */}
      {[0, 0.5, 1].map(ratio => (
        <line
          key={ratio}
          x1={padding.left}
          y1={padding.top + chartHeight * (1 - ratio)}
          x2={padding.left + chartWidth}
          y2={padding.top + chartHeight * (1 - ratio)}
          stroke="#374151"
          strokeWidth="1"
          strokeDasharray="4 4"
        />
      ))}

      {data.map((d, i) => {
        const barHeight = (d.value / maxValue) * chartHeight;
        const x = padding.left + (i * (chartWidth / data.length)) + (chartWidth / data.length - barWidth) / 2;
        const y = padding.top + chartHeight - barHeight;

        return (
          <g key={i}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              rx="3"
              fill={d.color}
              fillOpacity="0.8"
            />
            <text
              x={x + barWidth / 2}
              y={height - padding.bottom + 15}
              textAnchor="middle"
              fontSize="10"
              fill="#6b7280"
              transform={`rotate(-45, ${x + barWidth / 2}, ${height - padding.bottom + 15})`}
            >
              {d.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/**
 * Stat Card Component
 */
function StatCard({
  label,
  value,
  subValue,
  icon: Icon,
  color = 'cyan',
  trend
}: {
  label: string;
  value: string | number;
  subValue?: string;
  icon: React.ElementType;
  color?: string;
  trend?: 'up' | 'down' | 'neutral';
}) {
  const colorClasses: Record<string, string> = {
    cyan: 'from-cyan-950/50 to-cyan-900/30 border-cyan-700/30 text-cyan-400',
    purple: 'from-purple-950/50 to-purple-900/30 border-purple-700/30 text-purple-400',
    amber: 'from-amber-950/50 to-amber-900/30 border-amber-700/30 text-amber-400',
    rose: 'from-rose-950/50 to-rose-900/30 border-rose-700/30 text-rose-400',
    green: 'from-emerald-950/50 to-emerald-900/30 border-emerald-700/30 text-emerald-400'
  };

  return (
    <div className={`rounded-xl p-4 bg-gradient-to-br ${colorClasses[color] || colorClasses.cyan} border`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-400">{label}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          {subValue && <p className="text-xs text-gray-500 mt-1">{subValue}</p>}
        </div>
        <div className="p-2 rounded-lg bg-black/20">
          <Icon className="w-5 h-5" />
        </div>
      </div>
      {trend && (
        <div className={`text-xs mt-2 flex items-center gap-1 ${
          trend === 'up' ? 'text-emerald-400' :
          trend === 'down' ? 'text-rose-400' :
          'text-gray-400'
        }`}>
          {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'}
          {trend === 'up' ? 'Improving' : trend === 'down' ? 'Declining' : 'Stable'}
        </div>
      )}
    </div>
  );
}

/**
 * Main Analytics Dashboard Component
 */
export function AnalyticsDashboard({ checkIns, waves, onClose }: AnalyticsDashboardProps) {
  const [timeRange, setTimeRange] = useState<7 | 14 | 30>(7);
  const [selectedView, setSelectedView] = useState<'overview' | 'trends' | 'categories'>('overview');

  // Calculate analytics data
  const analytics = useMemo(() => {
    const dates = getDateRange(timeRange);
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Daily stats
    const dailyStats = dates.map(date => {
      const dayCheckIns = checkIns.filter(c => sameDay(new Date(c.slot), date));
      const completed = dayCheckIns.filter(c => c.done).length;
      const total = dayCheckIns.length;

      return {
        date,
        label: formatDate(date),
        completed,
        total,
        completionRate: total > 0 ? Math.round((completed / total) * 100) : 0
      };
    });

    // Category breakdown
    const categoryStats: Record<string, { total: number; completed: number }> = {};
    checkIns.forEach(c => {
      const date = new Date(c.slot);
      if (date >= dates[0] && date <= dates[dates.length - 1]) {
        if (!categoryStats[c.category]) {
          categoryStats[c.category] = { total: 0, completed: 0 };
        }
        categoryStats[c.category].total++;
        if (c.done) categoryStats[c.category].completed++;
      }
    });

    // Wave distribution
    const waveStats: Record<string, number> = {};
    checkIns.forEach(c => {
      const date = new Date(c.slot);
      if (date >= dates[0] && date <= dates[dates.length - 1] && c.waveId) {
        waveStats[c.waveId] = (waveStats[c.waveId] || 0) + 1;
      }
    });

    // Anchor completion rate
    const anchors = checkIns.filter(c => {
      const date = new Date(c.slot);
      return c.isAnchor && date >= dates[0] && date <= dates[dates.length - 1];
    });
    const completedAnchors = anchors.filter(c => c.done).length;

    // Overall stats
    const totalTasks = dailyStats.reduce((sum, d) => sum + d.total, 0);
    const completedTasks = dailyStats.reduce((sum, d) => sum + d.completed, 0);
    const avgCompletionRate = totalTasks > 0
      ? Math.round((completedTasks / totalTasks) * 100)
      : 0;

    // Trend calculation (compare first half to second half)
    const midpoint = Math.floor(dailyStats.length / 2);
    const firstHalf = dailyStats.slice(0, midpoint);
    const secondHalf = dailyStats.slice(midpoint);

    const firstHalfAvg = firstHalf.length > 0
      ? firstHalf.reduce((sum, d) => sum + d.completionRate, 0) / firstHalf.length
      : 0;
    const secondHalfAvg = secondHalf.length > 0
      ? secondHalf.reduce((sum, d) => sum + d.completionRate, 0) / secondHalf.length
      : 0;

    const trend: 'up' | 'down' | 'neutral' =
      secondHalfAvg > firstHalfAvg + 5 ? 'up' :
      secondHalfAvg < firstHalfAvg - 5 ? 'down' : 'neutral';

    return {
      dailyStats,
      categoryStats,
      waveStats,
      anchors: { total: anchors.length, completed: completedAnchors },
      totalTasks,
      completedTasks,
      avgCompletionRate,
      trend,
      avgTasksPerDay: Math.round(totalTasks / timeRange * 10) / 10
    };
  }, [checkIns, timeRange]);

  // Prepare chart data
  const completionChartData = analytics.dailyStats.map(d => ({
    label: d.label,
    value: d.completionRate
  }));

  const categoryChartData = Object.entries(analytics.categoryStats)
    .map(([category, stats]) => ({
      label: category,
      value: stats.total,
      color: CATEGORY_COLORS[category] || COLORS.gray
    }))
    .sort((a, b) => b.value - a.value);

  const waveChartData = waves
    .filter(w => analytics.waveStats[w.id])
    .map(w => ({
      label: w.name,
      value: analytics.waveStats[w.id] || 0,
      color: COLORS[w.color as keyof typeof COLORS] || COLORS.gray
    }));

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-4xl max-h-[90vh] bg-gray-950 border border-gray-800 rounded-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-cyan-400" />
            <div>
              <h2 className="text-xl font-light">Analytics Dashboard</h2>
              <p className="text-xs text-gray-500">Rhythm patterns and insights</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Time range selector */}
            <div className="flex rounded-lg overflow-hidden border border-gray-700">
              {([7, 14, 30] as const).map(days => (
                <button
                  key={days}
                  onClick={() => setTimeRange(days)}
                  className={`px-3 py-1.5 text-xs ${
                    timeRange === days
                      ? 'bg-cyan-600 text-white'
                      : 'bg-gray-900 text-gray-400 hover:bg-gray-800'
                  }`}
                >
                  {days}D
                </button>
              ))}
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* View Tabs */}
        <div className="flex items-center gap-1 p-2 border-b border-gray-800 bg-gray-900/30">
          {(['overview', 'trends', 'categories'] as const).map(view => (
            <button
              key={view}
              onClick={() => setSelectedView(view)}
              className={`px-4 py-2 rounded-lg text-sm capitalize ${
                selectedView === view
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              {view}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {selectedView === 'overview' && (
            <div className="space-y-6">
              {/* Stat Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard
                  label="Completion Rate"
                  value={`${analytics.avgCompletionRate}%`}
                  subValue={`${analytics.completedTasks}/${analytics.totalTasks} tasks`}
                  icon={TrendingUp}
                  color="cyan"
                  trend={analytics.trend}
                />
                <StatCard
                  label="Avg Tasks/Day"
                  value={analytics.avgTasksPerDay}
                  subValue={`Over ${timeRange} days`}
                  icon={Calendar}
                  color="purple"
                />
                <StatCard
                  label="Anchor Rate"
                  value={analytics.anchors.total > 0
                    ? `${Math.round((analytics.anchors.completed / analytics.anchors.total) * 100)}%`
                    : 'N/A'}
                  subValue={`${analytics.anchors.completed}/${analytics.anchors.total} anchors`}
                  icon={Zap}
                  color="amber"
                />
                <StatCard
                  label="Categories"
                  value={Object.keys(analytics.categoryStats).length}
                  subValue="Active categories"
                  icon={PieChart}
                  color="green"
                />
              </div>

              {/* Charts Row */}
              <div className="grid md:grid-cols-2 gap-6">
                {/* Completion Trend */}
                <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
                  <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-cyan-400" />
                    Completion Rate Trend
                  </h3>
                  <LineChart
                    data={completionChartData}
                    width={350}
                    height={180}
                    color={COLORS.cyan}
                    showArea
                  />
                </div>

                {/* Wave Distribution */}
                <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
                  <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
                    <PieChart className="w-4 h-4 text-purple-400" />
                    Wave Distribution
                  </h3>
                  <div className="flex items-center justify-center gap-6">
                    <DonutChart data={waveChartData} size={120} thickness={20} />
                    <div className="space-y-2">
                      {waveChartData.map(d => (
                        <div key={d.label} className="flex items-center gap-2 text-sm">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: d.color }}
                          />
                          <span className="text-gray-400">{d.label}</span>
                          <span className="text-gray-500">({d.value})</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedView === 'trends' && (
            <div className="space-y-6">
              {/* Daily Completion Chart */}
              <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-4">
                  Daily Completion Rate (%)
                </h3>
                <LineChart
                  data={completionChartData}
                  width={700}
                  height={250}
                  color={COLORS.cyan}
                  showArea
                />
              </div>

              {/* Daily Task Count */}
              <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-4">
                  Daily Task Volume
                </h3>
                <BarChart
                  data={analytics.dailyStats.map(d => ({
                    label: d.label,
                    value: d.total,
                    color: d.completionRate >= 80 ? COLORS.green :
                           d.completionRate >= 50 ? COLORS.amber : COLORS.rose
                  }))}
                  width={700}
                  height={200}
                />
              </div>
            </div>
          )}

          {selectedView === 'categories' && (
            <div className="space-y-6">
              {/* Category Breakdown */}
              <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-4">
                  Tasks by Category
                </h3>
                <BarChart
                  data={categoryChartData}
                  width={700}
                  height={300}
                  horizontal
                />
              </div>

              {/* Category Completion Rates */}
              <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-4">
                  Category Completion Rates
                </h3>
                <div className="space-y-3">
                  {Object.entries(analytics.categoryStats)
                    .sort((a, b) => b[1].total - a[1].total)
                    .map(([category, stats]) => {
                      const rate = stats.total > 0
                        ? Math.round((stats.completed / stats.total) * 100)
                        : 0;

                      return (
                        <div key={category} className="flex items-center gap-3">
                          <span className="w-24 text-sm text-gray-400">{category}</span>
                          <div className="flex-1 h-4 bg-gray-800 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{
                                width: `${rate}%`,
                                backgroundColor: CATEGORY_COLORS[category] || COLORS.gray
                              }}
                            />
                          </div>
                          <span className="w-12 text-sm text-gray-300 text-right">{rate}%</span>
                          <span className="w-20 text-xs text-gray-500">
                            {stats.completed}/{stats.total}
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AnalyticsDashboard;
