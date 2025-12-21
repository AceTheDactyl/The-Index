import React, { useState, useEffect } from 'react';
import {
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  Check,
  Play,
  SkipForward,
  RefreshCw,
  Sparkles,
  Target,
  Zap,
  Clock,
  AlertTriangle,
  Edit3,
  Save,
} from 'lucide-react';
import { userProfileService } from '../lib/userProfile';
import type { UserProfile } from '../lib/userProfile';
import { roadmapEngine } from '../lib/roadmapEngine';
import type { Roadmap, RoadmapStep } from '../lib/roadmapEngine';
import { glyphSystem, LIFE_DOMAINS, PHASE_DESCRIPTIONS } from '../lib/glyphSystem';
import type { WumboPhase, Glyph } from '../lib/glyphSystem';

interface RoadmapViewProps {
  domainId: string;
  onBack: () => void;
  onScheduleBeat?: (category: string, context: string) => void;
}

export const RoadmapView: React.FC<RoadmapViewProps> = ({
  domainId,
  onBack,
  onScheduleBeat,
}) => {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [roadmap, setRoadmap] = useState<Roadmap | null>(null);
  const domain = LIFE_DOMAINS.find(d => d.id === domainId);
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const [editingQuestions, setEditingQuestions] = useState(false);
  const [questionAnswers, setQuestionAnswers] = useState<Record<string, string>>({});
  const [currentFocus, setCurrentFocus] = useState('');
  const [targetOutcome, setTargetOutcome] = useState('');
  const [activePhase, setActivePhase] = useState<WumboPhase | null>(null);

  useEffect(() => {
    loadData();
  }, [domainId]);

  const loadData = async () => {
    const loadedProfile = await userProfileService.initialize();
    setProfile(loadedProfile);

    await roadmapEngine.initialize();
    const loadedRoadmap = roadmapEngine.getRoadmap(domainId);
    setRoadmap(loadedRoadmap || null);

    // Load domain question answers (new system)
    const domainData = loadedProfile.domainAnswers?.[domainId];
    if (domainData) {
      const answers: Record<string, string> = {};
      if (domainData.improvement) answers[`${domainId}-improvement`] = domainData.improvement;
      if (domainData.obstacle) answers[`${domainId}-obstacle`] = domainData.obstacle;
      if (domainData.emotion) answers[`${domainId}-emotion`] = domainData.emotion;
      if (domainData.vision) answers[`${domainId}-vision`] = domainData.vision;
      setQuestionAnswers(answers);
      setCurrentFocus(domainData.currentFocus || '');
      setTargetOutcome(domainData.targetOutcome || '');
    }
  };

  const toggleStep = (stepId: string) => {
    setExpandedSteps(prev => {
      const next = new Set(prev);
      if (next.has(stepId)) {
        next.delete(stepId);
      } else {
        next.add(stepId);
      }
      return next;
    });
  };

  const handleUpdateStepStatus = async (stepId: string, status: RoadmapStep['status']) => {
    if (!roadmap) return;
    await roadmapEngine.updateStepStatus(roadmap.id, stepId, status);
    const updated = roadmapEngine.getRoadmap(domainId);
    setRoadmap(updated || null);
  };

  const handleSaveQuestions = async () => {
    if (!profile) return;

    // Extract answers for this domain
    const domainAnswersToSave: {
      improvement?: string;
      obstacle?: string;
      emotion?: string;
      vision?: string;
      currentFocus?: string;
      targetOutcome?: string;
    } = {
      currentFocus,
      targetOutcome,
    };

    // Extract question answers
    for (const [key, answer] of Object.entries(questionAnswers)) {
      if (key.startsWith(`${domainId}-`)) {
        const field = key.replace(`${domainId}-`, '') as 'improvement' | 'obstacle' | 'emotion' | 'vision';
        domainAnswersToSave[field] = answer;
      }
    }

    // Save using the new domain answers method
    await userProfileService.saveDomainAnswers(domainId, domainAnswersToSave);

    setEditingQuestions(false);
    const updated = userProfileService.getProfile();
    if (updated) setProfile(updated);
  };

  const handleRefreshRoadmap = async () => {
    if (!profile) return;
    const metrics = profile.metricsHistory;
    await roadmapEngine.updateRoadmaps(profile, metrics);
    const updated = roadmapEngine.getRoadmap(domainId);
    setRoadmap(updated || null);
    await userProfileService.markRoadmapUpdated();
  };

  if (!domain) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-blue-950 flex items-center justify-center">
        <div className="text-red-400">Domain not found</div>
      </div>
    );
  }

  const glyphs = domain.glyphs.map(symbol =>
    glyphSystem.CORE_GLYPHS.find(g => g.symbol === symbol)
  ).filter(Boolean) as Glyph[];

  const phases = domain.phases.map(phase =>
    PHASE_DESCRIPTIONS.find(p => p.phase === phase)
  ).filter(Boolean);

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-blue-950 pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-black/60 border-b border-gray-800">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={onBack}
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back</span>
            </button>
            <button
              onClick={handleRefreshRoadmap}
              className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span className="text-sm">Refresh</span>
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-4 py-6">
        {/* Domain Header */}
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl p-6 mb-6">
          <div className="flex items-start gap-6">
            <div className="text-6xl">{domain.glyphs[0]}</div>
            <div className="flex-1">
              <h1 className="text-3xl font-light text-white mb-2">{domain.name}</h1>
              <p className="text-gray-400">{domain.description}</p>

              {/* Glyphs */}
              <div className="flex gap-2 mt-4">
                {domain.glyphs.map((symbol, i) => (
                  <span key={i} className="text-2xl" title={glyphs[i]?.name}>
                    {symbol}
                  </span>
                ))}
              </div>

              {/* Phases */}
              <div className="flex flex-wrap gap-2 mt-3">
                {phases.map(phase => (
                  <button
                    key={phase?.phase}
                    onClick={() => setActivePhase(activePhase === phase?.phase ? null : phase?.phase || null)}
                    className={`px-3 py-1 rounded-full text-xs ${
                      activePhase === phase?.phase
                        ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-500/50'
                        : 'bg-gray-800 text-gray-400 hover:text-white'
                    }`}
                  >
                    {phase?.name}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Phase Detail */}
          {activePhase && (
            <div className="mt-6 pt-6 border-t border-gray-800">
              {(() => {
                const phase = PHASE_DESCRIPTIONS.find(p => p.phase === activePhase);
                if (!phase) return null;
                return (
                  <div>
                    <h3 className="text-lg text-white mb-2">{phase.name} Phase</h3>
                    <p className="text-gray-400 mb-3">{phase.description}</p>
                    <p className="text-sm text-cyan-400 italic mb-4">"{phase.phenomenology}"</p>

                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm text-gray-500 mb-2">Rituals</h4>
                        <ul className="space-y-1">
                          {phase.rituals.map((ritual, i) => (
                            <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                              <span className="text-cyan-400">•</span>
                              {ritual}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="text-sm text-gray-500 mb-2">Warnings</h4>
                        <ul className="space-y-1">
                          {phase.warnings.map((warning, i) => (
                            <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                              <AlertTriangle className="w-3 h-3 text-yellow-400 flex-shrink-0 mt-0.5" />
                              {warning}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}
        </div>

        {/* Roadmap Questions */}
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-medium text-white flex items-center gap-2">
              <Target className="w-5 h-5 text-pink-400" />
              Your Focus
            </h2>
            {!editingQuestions ? (
              <button
                onClick={() => setEditingQuestions(true)}
                className="flex items-center gap-2 text-sm text-gray-400 hover:text-white"
              >
                <Edit3 className="w-4 h-4" />
                Edit
              </button>
            ) : (
              <div className="flex gap-2">
                <button
                  onClick={() => setEditingQuestions(false)}
                  className="px-3 py-1 text-sm text-gray-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveQuestions}
                  className="flex items-center gap-1 px-3 py-1 bg-cyan-500/20 text-cyan-300 rounded-lg text-sm"
                >
                  <Save className="w-3 h-3" />
                  Save
                </button>
              </div>
            )}
          </div>

          {/* Domain Questions */}
          <div className="space-y-4">
            {Object.entries(domain.questions).map(([key, question]) => (
              <div key={key}>
                <label className="text-sm text-gray-400 block mb-2">{question}</label>
                {editingQuestions ? (
                  <textarea
                    value={questionAnswers[`${domainId}-${key}`] || ''}
                    onChange={e => setQuestionAnswers({
                      ...questionAnswers,
                      [`${domainId}-${key}`]: e.target.value,
                    })}
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-2 text-white focus:border-cyan-500 focus:outline-none resize-none h-20"
                    placeholder="Your answer..."
                  />
                ) : (
                  <p className="text-gray-300 bg-gray-900/50 rounded-lg px-4 py-2 min-h-[50px]">
                    {questionAnswers[`${domainId}-${key}`] || (
                      <span className="text-gray-500 italic">Not answered yet</span>
                    )}
                  </p>
                )}
              </div>
            ))}
          </div>

          {/* Current Focus */}
          <div className="mt-6 pt-4 border-t border-gray-800">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-gray-400 block mb-2">Current Focus</label>
                {editingQuestions ? (
                  <input
                    type="text"
                    value={currentFocus}
                    onChange={e => setCurrentFocus(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-2 text-white focus:border-cyan-500 focus:outline-none"
                    placeholder="What are you focusing on now?"
                  />
                ) : (
                  <p className="text-cyan-300 font-medium">
                    {currentFocus || <span className="text-gray-500">Not set</span>}
                  </p>
                )}
              </div>
              <div>
                <label className="text-sm text-gray-400 block mb-2">Target Outcome</label>
                {editingQuestions ? (
                  <input
                    type="text"
                    value={targetOutcome}
                    onChange={e => setTargetOutcome(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-2 text-white focus:border-cyan-500 focus:outline-none"
                    placeholder="What do you want to achieve?"
                  />
                ) : (
                  <p className="text-purple-300 font-medium">
                    {targetOutcome || <span className="text-gray-500">Not set</span>}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Roadmap Steps */}
        {roadmap && (
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5 mb-6">
            <h2 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-400" />
              Your Roadmap
            </h2>

            <p className="text-sm text-gray-400 mb-4">{roadmap.description}</p>

            {/* Progress */}
            <div className="mb-6">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Progress</span>
                <span className="text-white">
                  {roadmap.steps.filter(s => s.status === 'completed').length}/{roadmap.steps.length} steps
                </span>
              </div>
              <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full transition-all"
                  style={{
                    width: `${(roadmap.steps.filter(s => s.status === 'completed').length / roadmap.steps.length) * 100}%`
                  }}
                />
              </div>
            </div>

            {/* Steps */}
            <div className="space-y-3">
              {roadmap.steps.map((step, index) => {
                const isExpanded = expandedSteps.has(step.id);
                const canStart = step.prerequisiteSteps.every(prereqId =>
                  roadmap.steps.find(s => s.id === prereqId)?.status === 'completed'
                );

                return (
                  <div
                    key={step.id}
                    className={`rounded-lg border transition-all ${
                      step.status === 'completed'
                        ? 'bg-green-500/10 border-green-500/30'
                        : step.status === 'active'
                        ? 'bg-cyan-500/10 border-cyan-500/30'
                        : 'bg-gray-900/50 border-gray-800'
                    }`}
                  >
                    <button
                      onClick={() => toggleStep(step.id)}
                      className="w-full p-4 flex items-center gap-4 text-left"
                    >
                      {/* Status Icon */}
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        step.status === 'completed'
                          ? 'bg-green-500/30 text-green-400'
                          : step.status === 'active'
                          ? 'bg-cyan-500/30 text-cyan-400'
                          : 'bg-gray-800 text-gray-500'
                      }`}>
                        {step.status === 'completed' ? (
                          <Check className="w-4 h-4" />
                        ) : step.status === 'active' ? (
                          <Play className="w-4 h-4" />
                        ) : (
                          <span className="text-sm">{index + 1}</span>
                        )}
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <h4 className={`font-medium ${
                          step.status === 'completed' ? 'text-green-300' :
                          step.status === 'active' ? 'text-cyan-300' :
                          'text-white'
                        }`}>
                          {step.title}
                        </h4>
                        <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {step.estimatedDuration}
                          </span>
                          <span className="capitalize">{step.actionType}</span>
                        </div>
                      </div>

                      {/* Expand Icon */}
                      {isExpanded ? (
                        <ChevronDown className="w-5 h-5 text-gray-500" />
                      ) : (
                        <ChevronRight className="w-5 h-5 text-gray-500" />
                      )}
                    </button>

                    {/* Expanded Content */}
                    {isExpanded && (
                      <div className="px-4 pb-4 pt-0 border-t border-gray-800/50 mt-2">
                        <p className="text-gray-400 text-sm mb-4">{step.description}</p>

                        {/* Metrics */}
                        {step.metrics.length > 0 && (
                          <div className="mb-4">
                            <h5 className="text-xs text-gray-500 mb-2">Target Metrics</h5>
                            <div className="flex flex-wrap gap-2">
                              {step.metrics.map((m, i) => (
                                <span
                                  key={i}
                                  className="px-2 py-1 bg-gray-800 rounded text-xs text-gray-300"
                                >
                                  {m.targetMetric}: {m.targetValue}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Actions */}
                        <div className="flex gap-2">
                          {step.status === 'pending' && canStart && (
                            <button
                              onClick={() => handleUpdateStepStatus(step.id, 'active')}
                              className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/20 text-cyan-300 rounded-lg text-sm hover:bg-cyan-500/30"
                            >
                              <Play className="w-3 h-3" />
                              Start
                            </button>
                          )}
                          {step.status === 'active' && (
                            <>
                              <button
                                onClick={() => handleUpdateStepStatus(step.id, 'completed')}
                                className="flex items-center gap-2 px-3 py-1.5 bg-green-500/20 text-green-300 rounded-lg text-sm hover:bg-green-500/30"
                              >
                                <Check className="w-3 h-3" />
                                Complete
                              </button>
                              <button
                                onClick={() => handleUpdateStepStatus(step.id, 'skipped')}
                                className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 text-gray-400 rounded-lg text-sm hover:bg-gray-700"
                              >
                                <SkipForward className="w-3 h-3" />
                                Skip
                              </button>
                            </>
                          )}
                          {step.status === 'pending' && !canStart && (
                            <span className="text-xs text-gray-500">
                              Complete prerequisite steps first
                            </span>
                          )}
                          {onScheduleBeat && step.status !== 'completed' && (
                            <button
                              onClick={() => onScheduleBeat(step.category, step.title)}
                              className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/20 text-purple-300 rounded-lg text-sm hover:bg-purple-500/30 ml-auto"
                            >
                              <Zap className="w-3 h-3" />
                              Schedule Beat
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Friction Sources */}
        {roadmap && roadmap.frictionSources.length > 0 && (
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5 mb-6">
            <h2 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
              Friction Sources
            </h2>
            <p className="text-sm text-gray-400 mb-4">
              These are the sources of friction identified in your {domain.name.toLowerCase()} domain.
              The roadmap is designed to address these systematically.
            </p>

            <div className="space-y-3">
              {roadmap.frictionSources.map(source => (
                <div
                  key={source.id}
                  className="p-3 bg-gray-900/50 rounded-lg border border-gray-800"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">{source.description}</span>
                    <span className={`text-sm px-2 py-0.5 rounded ${
                      source.intensity > 60 ? 'bg-red-500/20 text-red-300' :
                      source.intensity > 30 ? 'bg-yellow-500/20 text-yellow-300' :
                      'bg-green-500/20 text-green-300'
                    }`}>
                      {source.intensity}%
                    </span>
                  </div>
                  <div className="flex gap-2 mt-2">
                    {source.relatedBeats.map(beat => (
                      <span key={beat} className="text-xs text-gray-500">
                        #{beat}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Practices */}
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
          <h2 className="text-lg font-medium text-white mb-4">Recommended Practices</h2>
          <div className="grid md:grid-cols-2 gap-3">
            {domain.practices.map((practice, i) => (
              <div
                key={i}
                className="p-3 bg-gray-900/50 rounded-lg border border-gray-800 flex items-center gap-3"
              >
                <span className="text-lg">{domain.glyphs[i % domain.glyphs.length]}</span>
                <span className="text-gray-300">{practice}</span>
              </div>
            ))}
          </div>

          {/* Neural Regions */}
          <div className="mt-6 pt-4 border-t border-gray-800">
            <h3 className="text-sm text-gray-500 mb-3">Related Neural Regions (Ace Codex)</h3>
            <div className="flex flex-wrap gap-2">
              {domain.neuralRegions.map(region => (
                <span
                  key={region}
                  className="px-3 py-1 bg-purple-500/10 text-purple-300 rounded-full text-sm border border-purple-500/20"
                >
                  {region}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RoadmapView;
