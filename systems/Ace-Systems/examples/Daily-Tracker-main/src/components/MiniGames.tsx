// Mini Games Component - Interactive challenges (riddles, math, memory games)

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  X,
  Brain,
  Sparkles,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Trophy,
  Zap,
  Play,
  Eye,
  EyeOff,
  Clock,
  Lightbulb,
} from 'lucide-react';
import type { MiniGameData, ActiveChallenge } from '../lib/unifiedChallengeSystem';

interface MiniGamesProps {
  challenge: ActiveChallenge;
  onComplete: (score: number) => void;
  onClose: () => void;
}

export const MiniGames: React.FC<MiniGamesProps> = ({ challenge, onComplete, onClose }) => {
  const gameType = challenge.miniGameType;
  const gameData = challenge.miniGameData;

  if (!gameType || !gameData) {
    return (
      <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-gray-950 border border-gray-800 rounded-2xl p-6 text-center">
          <p className="text-gray-400">Game data not available</p>
          <button onClick={onClose} className="mt-4 px-4 py-2 bg-gray-800 rounded-lg">Close</button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="w-full max-w-lg bg-gray-950 border border-gray-800 rounded-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-800 bg-gradient-to-r from-purple-900/30 to-cyan-900/30">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{challenge.icon}</span>
            <div>
              <h3 className="text-lg font-medium text-white">{challenge.title}</h3>
              <p className="text-sm text-gray-400">{challenge.description}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg bg-gray-900/70 border border-gray-700 hover:bg-gray-800">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Game Content */}
        <div className="p-6">
          {gameType === 'riddle' && <RiddleGame data={gameData} onComplete={onComplete} />}
          {gameType === 'math_series' && <MathSeriesGame data={gameData} onComplete={onComplete} />}
          {gameType === 'color_sequence' && <ColorSequenceGame data={gameData} onComplete={onComplete} />}
          {gameType === 'card_memory' && <CardMemoryGame data={gameData} onComplete={onComplete} />}
          {gameType === 'word_scramble' && <WordScrambleGame data={gameData} onComplete={onComplete} />}
          {gameType === 'reaction_time' && <ReactionTimeGame data={gameData} onComplete={onComplete} />}
          {gameType === 'pattern_match' && <PatternMatchGame data={gameData} onComplete={onComplete} />}
          {gameType === 'speed_math' && <SpeedMathGame data={gameData} onComplete={onComplete} />}
        </div>

        {/* XP Reward */}
        <div className="px-6 pb-4">
          <div className="flex items-center justify-center gap-2 text-sm text-purple-300">
            <Sparkles className="w-4 h-4" />
            <span>+{challenge.xpReward} XP on completion</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// RIDDLE GAME
// ============================================================================

interface RiddleGameProps {
  data: MiniGameData;
  onComplete: (score: number) => void;
}

const RiddleGame: React.FC<RiddleGameProps> = ({ data, onComplete }) => {
  const [selected, setSelected] = useState<number | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [attempts, setAttempts] = useState(0);

  const handleSelect = (index: number) => {
    if (revealed) return;
    setSelected(index);
    setAttempts(prev => prev + 1);
    setRevealed(true);

    const isCorrect = index === data.correctAnswer;
    setTimeout(() => {
      if (isCorrect) {
        const score = Math.max(100 - (attempts * 20), 40);
        onComplete(score);
      }
    }, 1500);
  };

  const handleRetry = () => {
    setSelected(null);
    setRevealed(false);
  };

  const isCorrect = selected === data.correctAnswer;

  return (
    <div className="space-y-6">
      <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-800">
        <div className="flex items-start gap-3">
          <Brain className="w-6 h-6 text-purple-400 mt-1 flex-shrink-0" />
          <p className="text-lg text-gray-200 leading-relaxed">{data.question}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {data.options?.map((option, index) => (
          <button
            key={index}
            onClick={() => handleSelect(index)}
            disabled={revealed}
            className={`p-4 rounded-xl border text-left transition-all ${
              revealed && index === data.correctAnswer
                ? 'bg-emerald-900/30 border-emerald-500/50 text-emerald-300'
                : revealed && index === selected && !isCorrect
                ? 'bg-red-900/30 border-red-500/50 text-red-300'
                : selected === index
                ? 'bg-purple-900/30 border-purple-500/50 text-purple-300'
                : 'bg-gray-900/50 border-gray-700 hover:border-gray-600 text-gray-300'
            }`}
          >
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center text-sm font-medium">
                {String.fromCharCode(65 + index)}
              </span>
              <span>{option}</span>
              {revealed && index === data.correctAnswer && <CheckCircle2 className="w-5 h-5 text-emerald-400 ml-auto" />}
              {revealed && index === selected && !isCorrect && <XCircle className="w-5 h-5 text-red-400 ml-auto" />}
            </div>
          </button>
        ))}
      </div>

      {revealed && !isCorrect && (
        <button
          onClick={handleRetry}
          className="w-full py-3 rounded-xl bg-purple-600/20 border border-purple-500/50 text-purple-300 flex items-center justify-center gap-2 hover:bg-purple-600/30 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          Try Again
        </button>
      )}

      {revealed && isCorrect && (
        <div className="text-center p-4 bg-emerald-900/20 rounded-xl border border-emerald-500/30">
          <Trophy className="w-8 h-8 text-amber-400 mx-auto mb-2" />
          <p className="text-emerald-300 font-medium">Correct!</p>
          <p className="text-sm text-gray-400 mt-1">Challenge completed</p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// MATH SERIES GAME
// ============================================================================

interface MathSeriesGameProps {
  data: MiniGameData;
  onComplete: (score: number) => void;
}

const MathSeriesGame: React.FC<MathSeriesGameProps> = ({ data, onComplete }) => {
  const [answer, setAnswer] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [attempts, setAttempts] = useState(0);

  const handleSubmit = () => {
    const userAnswer = parseFloat(answer);
    setAttempts(prev => prev + 1);
    setSubmitted(true);

    if (userAnswer === data.answer) {
      const score = Math.max(100 - (attempts * 15), 50);
      setTimeout(() => onComplete(score), 1500);
    }
  };

  const handleRetry = () => {
    setAnswer('');
    setSubmitted(false);
  };

  const isCorrect = parseFloat(answer) === data.answer;

  return (
    <div className="space-y-6">
      <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-800">
        <p className="text-sm text-gray-400 mb-3">Find the next number in the sequence:</p>
        <div className="flex items-center gap-3 flex-wrap">
          {data.series?.map((num, index) => (
            <div key={index} className="px-4 py-2 bg-gray-800 rounded-lg text-xl font-mono text-cyan-300">
              {num}
            </div>
          ))}
          <div className="px-4 py-2 bg-purple-900/30 border-2 border-dashed border-purple-500/50 rounded-lg text-xl font-mono text-purple-300">
            ?
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <input
          type="number"
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          placeholder="Your answer..."
          disabled={submitted && isCorrect}
          className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-xl text-white text-center text-xl font-mono focus:border-purple-500 focus:outline-none"
          onKeyDown={(e) => e.key === 'Enter' && !submitted && handleSubmit()}
        />

        {!submitted ? (
          <button
            onClick={handleSubmit}
            disabled={!answer}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-purple-600 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Submit Answer
          </button>
        ) : isCorrect ? (
          <div className="text-center p-4 bg-emerald-900/20 rounded-xl border border-emerald-500/30">
            <Trophy className="w-8 h-8 text-amber-400 mx-auto mb-2" />
            <p className="text-emerald-300 font-medium">Correct! The answer is {data.answer}</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-center p-3 bg-red-900/20 rounded-xl border border-red-500/30">
              <p className="text-red-300">Not quite right. Try again!</p>
            </div>
            <button
              onClick={handleRetry}
              className="w-full py-3 rounded-xl bg-purple-600/20 border border-purple-500/50 text-purple-300 flex items-center justify-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// COLOR SEQUENCE GAME
// ============================================================================

interface ColorSequenceGameProps {
  data: MiniGameData;
  onComplete: (score: number) => void;
}

const ColorSequenceGame: React.FC<ColorSequenceGameProps> = ({ data, onComplete }) => {
  const [phase, setPhase] = useState<'watch' | 'input' | 'result'>('watch');
  const [showIndex, setShowIndex] = useState(0);
  const [userSequence, setUserSequence] = useState<string[]>([]);
  const [isCorrect, setIsCorrect] = useState(false);

  const colors = ['#EF4444', '#F59E0B', '#22C55E', '#3B82F6', '#8B5CF6', '#EC4899'];
  const sequence = data.sequence || [];

  useEffect(() => {
    if (phase === 'watch') {
      const timer = setInterval(() => {
        setShowIndex(prev => {
          if (prev >= sequence.length) {
            clearInterval(timer);
            setTimeout(() => {
              setPhase('input');
              setShowIndex(-1);
            }, 500);
            return prev;
          }
          return prev + 1;
        });
      }, 800);

      return () => clearInterval(timer);
    }
  }, [phase, sequence.length]);

  const handleColorClick = (color: string) => {
    if (phase !== 'input') return;

    const newSequence = [...userSequence, color];
    setUserSequence(newSequence);

    if (newSequence.length === sequence.length) {
      const correct = newSequence.every((c, i) => c === sequence[i]);
      setIsCorrect(correct);
      setPhase('result');

      if (correct) {
        const score = 100;
        setTimeout(() => onComplete(score), 1500);
      }
    }
  };

  const handleRestart = () => {
    setPhase('watch');
    setShowIndex(0);
    setUserSequence([]);
    setIsCorrect(false);
  };

  return (
    <div className="space-y-6">
      {/* Instructions */}
      <div className="text-center">
        {phase === 'watch' && (
          <div className="flex items-center justify-center gap-2 text-cyan-300">
            <Eye className="w-5 h-5" />
            <span>Watch the sequence carefully...</span>
          </div>
        )}
        {phase === 'input' && (
          <div className="flex items-center justify-center gap-2 text-purple-300">
            <Zap className="w-5 h-5" />
            <span>Now repeat the sequence!</span>
          </div>
        )}
      </div>

      {/* Display Area */}
      <div className="flex justify-center">
        <div
          className={`w-32 h-32 rounded-2xl border-4 transition-all duration-300 ${
            phase === 'watch' && showIndex < sequence.length
              ? 'border-white/50'
              : 'border-gray-700'
          }`}
          style={{
            backgroundColor: phase === 'watch' && showIndex < sequence.length
              ? sequence[showIndex]
              : '#1f2937',
          }}
        />
      </div>

      {/* Progress Indicators */}
      <div className="flex justify-center gap-2">
        {sequence.map((_, index) => (
          <div
            key={index}
            className={`w-3 h-3 rounded-full transition-all ${
              phase === 'watch' && index < showIndex
                ? 'bg-cyan-500'
                : phase === 'input' && index < userSequence.length
                ? userSequence[index] === sequence[index] ? 'bg-emerald-500' : 'bg-red-500'
                : 'bg-gray-700'
            }`}
          />
        ))}
      </div>

      {/* Color Buttons */}
      {phase === 'input' && (
        <div className="grid grid-cols-3 gap-3">
          {colors.map(color => (
            <button
              key={color}
              onClick={() => handleColorClick(color)}
              className="h-16 rounded-xl border-2 border-gray-700 hover:border-white/50 transition-all active:scale-95"
              style={{ backgroundColor: color }}
            />
          ))}
        </div>
      )}

      {/* Result */}
      {phase === 'result' && (
        <div className={`text-center p-4 rounded-xl border ${
          isCorrect
            ? 'bg-emerald-900/20 border-emerald-500/30'
            : 'bg-red-900/20 border-red-500/30'
        }`}>
          {isCorrect ? (
            <>
              <Trophy className="w-8 h-8 text-amber-400 mx-auto mb-2" />
              <p className="text-emerald-300 font-medium">Perfect sequence!</p>
            </>
          ) : (
            <>
              <XCircle className="w-8 h-8 text-red-400 mx-auto mb-2" />
              <p className="text-red-300 font-medium">Wrong sequence</p>
              <button
                onClick={handleRestart}
                className="mt-3 px-4 py-2 rounded-lg bg-purple-600/20 border border-purple-500/50 text-purple-300 flex items-center gap-2 mx-auto"
              >
                <RotateCcw className="w-4 h-4" />
                Try Again
              </button>
            </>
          )}
        </div>
      )}

      {phase === 'watch' && (
        <p className="text-center text-sm text-gray-500">
          Sequence {Math.min(showIndex + 1, sequence.length)} of {sequence.length}
        </p>
      )}
    </div>
  );
};

// ============================================================================
// CARD MEMORY GAME
// ============================================================================

interface CardMemoryGameProps {
  data: MiniGameData;
  onComplete: (score: number) => void;
}

interface MemoryCard {
  id: number;
  symbol: string;
  isFlipped: boolean;
  isMatched: boolean;
}

const CardMemoryGame: React.FC<CardMemoryGameProps> = ({ data, onComplete }) => {
  const pairs = data.pairs || 6;
  const symbols = data.cardSymbols || ['🌟', '🔥', '💎', '🌙', '⚡', '🎯'];

  const [cards, setCards] = useState<MemoryCard[]>([]);
  const [flippedCards, setFlippedCards] = useState<number[]>([]);
  const [moves, setMoves] = useState(0);
  const [matches, setMatches] = useState(0);
  const [gameStarted, setGameStarted] = useState(false);
  const [showPreview, setShowPreview] = useState(true);
  const lockRef = useRef(false);

  useEffect(() => {
    const cardSymbols = symbols.slice(0, pairs);
    const cardPairs = [...cardSymbols, ...cardSymbols]
      .sort(() => Math.random() - 0.5)
      .map((symbol, index) => ({
        id: index,
        symbol,
        isFlipped: true,
        isMatched: false,
      }));

    setCards(cardPairs);

    const timer = setTimeout(() => {
      setCards(prev => prev.map(card => ({ ...card, isFlipped: false })));
      setShowPreview(false);
      setGameStarted(true);
    }, 3000);

    return () => clearTimeout(timer);
  }, [pairs, symbols]);

  const handleCardClick = useCallback((cardId: number) => {
    if (!gameStarted || lockRef.current) return;

    const card = cards.find(c => c.id === cardId);
    if (!card || card.isFlipped || card.isMatched) return;

    const newFlipped = [...flippedCards, cardId];
    setFlippedCards(newFlipped);
    setCards(prev => prev.map(c => c.id === cardId ? { ...c, isFlipped: true } : c));

    if (newFlipped.length === 2) {
      lockRef.current = true;
      setMoves(prev => prev + 1);

      const [first, second] = newFlipped;
      const firstCard = cards.find(c => c.id === first);
      const secondCard = cards.find(c => c.id === second);

      if (firstCard?.symbol === secondCard?.symbol) {
        setTimeout(() => {
          setCards(prev => prev.map(c =>
            c.id === first || c.id === second ? { ...c, isMatched: true } : c
          ));
          setMatches(prev => {
            const newMatches = prev + 1;
            if (newMatches === pairs) {
              const baseScore = 100;
              const movePenalty = Math.max(0, (moves - pairs) * 5);
              const score = Math.max(baseScore - movePenalty, 50);
              setTimeout(() => onComplete(score), 500);
            }
            return newMatches;
          });
          setFlippedCards([]);
          lockRef.current = false;
        }, 500);
      } else {
        setTimeout(() => {
          setCards(prev => prev.map(c =>
            c.id === first || c.id === second ? { ...c, isFlipped: false } : c
          ));
          setFlippedCards([]);
          lockRef.current = false;
        }, 1000);
      }
    }
  }, [cards, flippedCards, gameStarted, moves, pairs, onComplete]);

  const handleRestart = () => {
    setMoves(0);
    setMatches(0);
    setFlippedCards([]);
    setShowPreview(true);
    setGameStarted(false);
    lockRef.current = false;

    const cardSymbols = symbols.slice(0, pairs);
    const cardPairs = [...cardSymbols, ...cardSymbols]
      .sort(() => Math.random() - 0.5)
      .map((symbol, index) => ({
        id: index,
        symbol,
        isFlipped: true,
        isMatched: false,
      }));

    setCards(cardPairs);

    setTimeout(() => {
      setCards(prev => prev.map(card => ({ ...card, isFlipped: false })));
      setShowPreview(false);
      setGameStarted(true);
    }, 3000);
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center px-2">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Play className="w-4 h-4" />
            <span>Moves: {moves}</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-purple-300">
            <Sparkles className="w-4 h-4" />
            <span>Matches: {matches}/{pairs}</span>
          </div>
        </div>
        <button
          onClick={handleRestart}
          className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700"
          title="Restart"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {showPreview && (
        <div className="text-center p-3 bg-cyan-900/20 rounded-xl border border-cyan-500/30">
          <div className="flex items-center justify-center gap-2 text-cyan-300">
            <Eye className="w-5 h-5" />
            <span>Memorize the cards!</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-4 gap-2">
        {cards.map(card => (
          <button
            key={card.id}
            onClick={() => handleCardClick(card.id)}
            disabled={!gameStarted || card.isFlipped || card.isMatched}
            className={`aspect-square rounded-xl text-2xl transition-all transform ${
              card.isFlipped || card.isMatched
                ? 'bg-gradient-to-br from-purple-600/30 to-cyan-600/30 border-purple-500/50 scale-100'
                : 'bg-gray-800 border-gray-700 hover:border-gray-600 hover:scale-105'
            } border-2 ${
              card.isMatched ? 'opacity-60' : ''
            }`}
          >
            {(card.isFlipped || card.isMatched) && (
              <span className="block">{card.symbol}</span>
            )}
            {!card.isFlipped && !card.isMatched && (
              <EyeOff className="w-5 h-5 mx-auto text-gray-600" />
            )}
          </button>
        ))}
      </div>

      {matches === pairs && (
        <div className="text-center p-4 bg-emerald-900/20 rounded-xl border border-emerald-500/30">
          <Trophy className="w-8 h-8 text-amber-400 mx-auto mb-2" />
          <p className="text-emerald-300 font-medium">All pairs found!</p>
          <p className="text-sm text-gray-400 mt-1">Completed in {moves} moves</p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// WORD SCRAMBLE GAME
// ============================================================================

interface WordScrambleGameProps {
  data: MiniGameData;
  onComplete: (score: number) => void;
}

const WordScrambleGame: React.FC<WordScrambleGameProps> = ({ data, onComplete }) => {
  const word = data.word || 'MINDFUL';
  const hint = data.hint || 'Being present in the moment';

  const [scrambled] = useState(() => {
    const chars = word.split('');
    for (let i = chars.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [chars[i], chars[j]] = [chars[j], chars[i]];
    }
    return chars.join('');
  });

  const [guess, setGuess] = useState('');
  const [attempts, setAttempts] = useState(0);
  const [showHint, setShowHint] = useState(false);
  const [completed, setCompleted] = useState(false);

  const handleSubmit = () => {
    setAttempts(prev => prev + 1);
    if (guess.toUpperCase() === word.toUpperCase()) {
      setCompleted(true);
      const score = Math.max(100 - (attempts * 10) - (showHint ? 20 : 0), 40);
      setTimeout(() => onComplete(score), 1500);
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <p className="text-sm text-gray-400 mb-3">Unscramble the word:</p>
        <div className="flex justify-center gap-2">
          {scrambled.split('').map((letter, i) => (
            <div
              key={i}
              className="w-12 h-12 bg-purple-900/30 border border-purple-500/50 rounded-lg flex items-center justify-center text-2xl font-bold text-purple-300"
            >
              {letter}
            </div>
          ))}
        </div>
      </div>

      {!showHint && !completed && (
        <button
          onClick={() => setShowHint(true)}
          className="w-full py-2 rounded-lg bg-amber-900/20 border border-amber-500/30 text-amber-300 flex items-center justify-center gap-2"
        >
          <Lightbulb className="w-4 h-4" />
          Show Hint (-20 points)
        </button>
      )}

      {showHint && (
        <div className="p-3 bg-amber-900/20 rounded-lg border border-amber-500/30">
          <p className="text-sm text-amber-300">💡 Hint: {hint}</p>
        </div>
      )}

      {!completed ? (
        <div className="space-y-3">
          <input
            type="text"
            value={guess}
            onChange={(e) => setGuess(e.target.value.toUpperCase())}
            placeholder="Your answer..."
            maxLength={word.length}
            className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-xl text-white text-center text-xl font-mono uppercase focus:border-purple-500 focus:outline-none"
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
          />
          <button
            onClick={handleSubmit}
            disabled={guess.length !== word.length}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-600 to-cyan-600 text-white font-medium disabled:opacity-50"
          >
            Check Answer
          </button>
          {attempts > 0 && !completed && (
            <p className="text-center text-sm text-red-400">Not correct, try again!</p>
          )}
        </div>
      ) : (
        <div className="text-center p-4 bg-emerald-900/20 rounded-xl border border-emerald-500/30">
          <Trophy className="w-8 h-8 text-amber-400 mx-auto mb-2" />
          <p className="text-emerald-300 font-medium">Correct! The word was {word}</p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// REACTION TIME GAME
// ============================================================================

interface ReactionTimeGameProps {
  data: MiniGameData;
  onComplete: (score: number) => void;
}

const ReactionTimeGame: React.FC<ReactionTimeGameProps> = ({ data, onComplete }) => {
  const targetTime = data.targetTime || 3;
  const [phase, setPhase] = useState<'waiting' | 'ready' | 'click' | 'result'>('waiting');
  const [reactionTime, setReactionTime] = useState<number | null>(null);
  const [attempts, setAttempts] = useState(0);
  const startTimeRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const startGame = () => {
    setPhase('ready');
    setReactionTime(null);
    setAttempts(prev => prev + 1);

    const delay = Math.random() * 3000 + 2000; // 2-5 seconds
    timerRef.current = setTimeout(() => {
      setPhase('click');
      startTimeRef.current = Date.now();
    }, delay);
  };

  const handleClick = () => {
    if (phase === 'ready') {
      // Clicked too early
      if (timerRef.current) clearTimeout(timerRef.current);
      setReactionTime(-1);
      setPhase('result');
    } else if (phase === 'click') {
      const time = Date.now() - startTimeRef.current;
      setReactionTime(time);
      setPhase('result');

      if (time < targetTime * 1000) {
        const score = Math.max(100 - Math.floor(time / 10), 50);
        setTimeout(() => onComplete(score), 1500);
      }
    }
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return (
    <div className="space-y-6">
      <div className="text-center">
        <p className="text-sm text-gray-400 mb-2">Click as fast as you can when the screen turns green!</p>
        <p className="text-xs text-purple-300">Target: under {targetTime * 1000}ms</p>
      </div>

      <button
        onClick={phase === 'waiting' || phase === 'result' ? startGame : handleClick}
        className={`w-full h-40 rounded-2xl text-xl font-medium transition-all ${
          phase === 'waiting' ? 'bg-gray-800 text-gray-400 hover:bg-gray-700' :
          phase === 'ready' ? 'bg-red-600 text-white' :
          phase === 'click' ? 'bg-emerald-500 text-white animate-pulse' :
          'bg-gray-800 text-gray-400'
        }`}
      >
        {phase === 'waiting' && 'Click to Start'}
        {phase === 'ready' && 'Wait for green...'}
        {phase === 'click' && 'CLICK NOW!'}
        {phase === 'result' && (
          reactionTime === -1 ? 'Too early! Click to retry' :
          reactionTime && reactionTime < targetTime * 1000 ? `${reactionTime}ms - Great!` :
          `${reactionTime}ms - Try again`
        )}
      </button>

      {phase === 'result' && reactionTime && reactionTime > 0 && reactionTime < targetTime * 1000 && (
        <div className="text-center p-4 bg-emerald-900/20 rounded-xl border border-emerald-500/30">
          <Trophy className="w-8 h-8 text-amber-400 mx-auto mb-2" />
          <p className="text-emerald-300 font-medium">Fast reflexes!</p>
        </div>
      )}

      <div className="flex justify-center gap-4 text-sm text-gray-500">
        <span>Attempts: {attempts}</span>
        {reactionTime && reactionTime > 0 && <span>Last: {reactionTime}ms</span>}
      </div>
    </div>
  );
};

// ============================================================================
// PATTERN MATCH GAME
// ============================================================================

interface PatternMatchGameProps {
  data: MiniGameData;
  onComplete: (score: number) => void;
}

const PatternMatchGame: React.FC<PatternMatchGameProps> = ({ data, onComplete }) => {
  const gridSize = data.gridSize || 3;
  const [pattern, setPattern] = useState<boolean[]>([]);
  const [userPattern, setUserPattern] = useState<boolean[]>([]);
  const [phase, setPhase] = useState<'show' | 'input' | 'result'>('show');
  const [score, setScore] = useState(0);
  const [round, setRound] = useState(1);

  useEffect(() => {
    generatePattern();
  }, []);

  const generatePattern = () => {
    const newPattern = Array(gridSize * gridSize).fill(false);
    const activeCount = Math.min(round + 2, gridSize * gridSize - 1);
    const indices = [...Array(gridSize * gridSize).keys()];

    for (let i = 0; i < activeCount; i++) {
      const randomIndex = Math.floor(Math.random() * indices.length);
      newPattern[indices[randomIndex]] = true;
      indices.splice(randomIndex, 1);
    }

    setPattern(newPattern);
    setUserPattern(Array(gridSize * gridSize).fill(false));
    setPhase('show');

    setTimeout(() => setPhase('input'), 2000);
  };

  const handleCellClick = (index: number) => {
    if (phase !== 'input') return;

    const newUserPattern = [...userPattern];
    newUserPattern[index] = !newUserPattern[index];
    setUserPattern(newUserPattern);
  };

  const checkPattern = () => {
    const correct = pattern.every((val, i) => val === userPattern[i]);
    setPhase('result');

    if (correct) {
      const newScore = score + round * 10;
      setScore(newScore);

      if (round >= 5) {
        setTimeout(() => onComplete(Math.min(100, newScore)), 1500);
      } else {
        setTimeout(() => {
          setRound(prev => prev + 1);
          generatePattern();
        }, 1500);
      }
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <span className="text-sm text-gray-400">Round {round}/5</span>
        <span className="text-sm text-purple-300">Score: {score}</span>
      </div>

      <div className="text-center">
        {phase === 'show' && <p className="text-cyan-300">Memorize the pattern...</p>}
        {phase === 'input' && <p className="text-purple-300">Recreate the pattern!</p>}
      </div>

      <div
        className="grid gap-2 mx-auto"
        style={{
          gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
          maxWidth: `${gridSize * 60}px`
        }}
      >
        {(phase === 'show' ? pattern : userPattern).map((active, index) => (
          <button
            key={index}
            onClick={() => handleCellClick(index)}
            disabled={phase !== 'input'}
            className={`aspect-square rounded-lg transition-all ${
              active
                ? 'bg-purple-500 border-purple-400'
                : 'bg-gray-800 border-gray-700'
            } border-2 ${phase === 'input' ? 'hover:border-purple-400' : ''}`}
          />
        ))}
      </div>

      {phase === 'input' && (
        <button
          onClick={checkPattern}
          className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-600 to-cyan-600 text-white font-medium"
        >
          Check Pattern
        </button>
      )}

      {phase === 'result' && (
        <div className={`text-center p-4 rounded-xl border ${
          pattern.every((val, i) => val === userPattern[i])
            ? 'bg-emerald-900/20 border-emerald-500/30'
            : 'bg-red-900/20 border-red-500/30'
        }`}>
          {pattern.every((val, i) => val === userPattern[i]) ? (
            <>
              <CheckCircle2 className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
              <p className="text-emerald-300 font-medium">
                {round >= 5 ? 'All rounds complete!' : 'Correct! Next round...'}
              </p>
            </>
          ) : (
            <>
              <XCircle className="w-8 h-8 text-red-400 mx-auto mb-2" />
              <p className="text-red-300 font-medium">Wrong pattern</p>
              <button
                onClick={() => {
                  setScore(0);
                  setRound(1);
                  generatePattern();
                }}
                className="mt-3 px-4 py-2 rounded-lg bg-purple-600/20 border border-purple-500/50 text-purple-300"
              >
                Try Again
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// SPEED MATH GAME
// ============================================================================

interface SpeedMathGameProps {
  data: MiniGameData;
  onComplete: (score: number) => void;
}

const SpeedMathGame: React.FC<SpeedMathGameProps> = ({ data, onComplete }) => {
  const totalProblems = data.problemCount || 5;
  const timeLimit = data.timeLimit || 30;

  const [currentProblem, setCurrentProblem] = useState({ a: 0, b: 0, op: '+', answer: 0 });
  const [userAnswer, setUserAnswer] = useState('');
  const [problemIndex, setProblemIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(timeLimit);
  const [gameOver, setGameOver] = useState(false);

  const generateProblem = useCallback(() => {
    const operations = ['+', '-', '×'];
    const op = operations[Math.floor(Math.random() * operations.length)];
    let a: number, b: number, answer: number;

    switch (op) {
      case '+':
        a = Math.floor(Math.random() * 50) + 1;
        b = Math.floor(Math.random() * 50) + 1;
        answer = a + b;
        break;
      case '-':
        a = Math.floor(Math.random() * 50) + 20;
        b = Math.floor(Math.random() * (a - 1)) + 1;
        answer = a - b;
        break;
      case '×':
        a = Math.floor(Math.random() * 12) + 1;
        b = Math.floor(Math.random() * 12) + 1;
        answer = a * b;
        break;
      default:
        a = 1; b = 1; answer = 2;
    }

    return { a, b, op, answer };
  }, []);

  useEffect(() => {
    setCurrentProblem(generateProblem());
  }, [generateProblem]);

  useEffect(() => {
    if (gameOver || timeLeft <= 0) return;

    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          setGameOver(true);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [gameOver]);

  useEffect(() => {
    if (gameOver && score > 0) {
      const finalScore = Math.min(100, Math.floor((score / totalProblems) * 100));
      setTimeout(() => onComplete(finalScore), 1500);
    }
  }, [gameOver, score, totalProblems, onComplete]);

  const handleSubmit = () => {
    const isCorrect = parseInt(userAnswer) === currentProblem.answer;

    if (isCorrect) {
      setScore(prev => prev + 1);
    }

    if (problemIndex + 1 >= totalProblems) {
      setGameOver(true);
    } else {
      setProblemIndex(prev => prev + 1);
      setCurrentProblem(generateProblem());
      setUserAnswer('');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-amber-400" />
          <span className={`text-lg font-mono ${timeLeft <= 10 ? 'text-red-400' : 'text-gray-300'}`}>
            {timeLeft}s
          </span>
        </div>
        <span className="text-sm text-gray-400">
          Problem {problemIndex + 1}/{totalProblems}
        </span>
        <span className="text-sm text-emerald-400">
          Score: {score}
        </span>
      </div>

      {!gameOver ? (
        <>
          <div className="text-center p-6 bg-gray-900/50 rounded-xl">
            <p className="text-4xl font-bold text-white">
              {currentProblem.a} {currentProblem.op} {currentProblem.b} = ?
            </p>
          </div>

          <div className="space-y-3">
            <input
              type="number"
              value={userAnswer}
              onChange={(e) => setUserAnswer(e.target.value)}
              placeholder="Answer..."
              autoFocus
              className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-xl text-white text-center text-2xl font-mono focus:border-purple-500 focus:outline-none"
              onKeyDown={(e) => e.key === 'Enter' && userAnswer && handleSubmit()}
            />
            <button
              onClick={handleSubmit}
              disabled={!userAnswer}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-600 to-cyan-600 text-white font-medium disabled:opacity-50"
            >
              Submit
            </button>
          </div>
        </>
      ) : (
        <div className="text-center p-6 bg-gray-900/50 rounded-xl">
          <Trophy className="w-12 h-12 text-amber-400 mx-auto mb-3" />
          <p className="text-2xl font-bold text-white mb-2">
            {score}/{totalProblems} Correct!
          </p>
          <p className="text-gray-400">
            {score === totalProblems ? 'Perfect score!' :
             score >= totalProblems * 0.8 ? 'Great job!' :
             score >= totalProblems * 0.5 ? 'Good effort!' : 'Keep practicing!'}
          </p>
        </div>
      )}
    </div>
  );
};

export default MiniGames;
