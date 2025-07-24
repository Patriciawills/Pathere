"""
Lateral Thinking Module - Phase 3.1.1
Enables creative problem-solving through unexpected connections and novel approaches
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

class ThinkingPattern(Enum):
    ANALOGICAL = "analogical"
    REVERSE = "reverse"
    RANDOM_STIMULUS = "random_stimulus"
    ASSUMPTION_CHALLENGING = "assumption_challenging"
    PERSPECTIVE_SHIFTING = "perspective_shifting"
    COMBINATION = "combination"

@dataclass
class LateralInsight:
    id: str
    problem: str
    insight: str
    thinking_pattern: ThinkingPattern
    confidence: float
    novelty_score: float
    connections_made: List[str]
    evidence: List[str]
    applications: List[str]
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['thinking_pattern'] = self.thinking_pattern.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class CreativeSolution:
    id: str
    original_problem: str
    solution: str
    approach_used: str
    creativity_score: float
    feasibility_score: float
    uniqueness_score: float
    steps_taken: List[str]
    inspiration_sources: List[str]
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class LateralThinkingModule:
    """
    Advanced lateral thinking capabilities for creative problem-solving
    """
    
    def __init__(self):
        self.insights_history: List[LateralInsight] = []
        self.solutions_history: List[CreativeSolution] = []
        self.thinking_patterns_used: Dict[str, int] = {}
        self.random_stimuli_database = self._initialize_stimuli_database()
        self.analogy_domains = self._initialize_analogy_domains()
        
    def _initialize_stimuli_database(self) -> List[str]:
        """Initialize database of random stimuli for creative thinking"""
        return [
            "butterfly", "clockwork", "mirror", "storm", "seed", "river", "mountain", "fire",
            "dance", "music", "puzzle", "bridge", "ocean", "star", "forest", "key",
            "telescope", "compass", "feather", "crystal", "wind", "shadow", "light", "spiral",
            "web", "cascade", "echo", "rhythm", "texture", "harmony", "balance", "flow",
            "growth", "transformation", "connection", "emergence", "resonance", "synthesis"
        ]
    
    def _initialize_analogy_domains(self) -> Dict[str, List[str]]:
        """Initialize domains for analogical thinking"""
        return {
            "nature": ["ecosystem", "evolution", "symbiosis", "adaptation", "growth", "cycles"],
            "mechanics": ["leverage", "friction", "momentum", "equilibrium", "resonance", "efficiency"],
            "architecture": ["foundation", "support", "structure", "flow", "integration", "harmony"],
            "music": ["rhythm", "harmony", "composition", "improvisation", "resonance", "dynamics"],
            "cooking": ["ingredients", "process", "timing", "temperature", "blending", "seasoning"],
            "sports": ["strategy", "teamwork", "practice", "timing", "adaptation", "performance"],
            "art": ["composition", "color", "perspective", "expression", "creativity", "interpretation"]
        }
    
    async def generate_lateral_insight(self, problem: str, context: Dict[str, Any] = None) -> LateralInsight:
        """Generate creative insights using lateral thinking patterns"""
        
        # Select thinking pattern based on problem type and context
        pattern = self._select_thinking_pattern(problem, context)
        
        # Generate insight based on selected pattern
        insight_data = await self._apply_thinking_pattern(problem, pattern, context)
        
        insight = LateralInsight(
            id=str(uuid.uuid4()),
            problem=problem,
            insight=insight_data["insight"],
            thinking_pattern=pattern,
            confidence=insight_data["confidence"],
            novelty_score=insight_data["novelty_score"],
            connections_made=insight_data["connections"],
            evidence=insight_data["evidence"],
            applications=insight_data["applications"],
            timestamp=datetime.utcnow()
        )
        
        self.insights_history.append(insight)
        self.thinking_patterns_used[pattern.value] = self.thinking_patterns_used.get(pattern.value, 0) + 1
        
        return insight
    
    def _select_thinking_pattern(self, problem: str, context: Dict[str, Any] = None) -> ThinkingPattern:
        """Select most appropriate thinking pattern for the problem"""
        
        problem_lower = problem.lower()
        
        # Pattern selection based on problem characteristics
        if "assumption" in problem_lower or "traditional" in problem_lower:
            return ThinkingPattern.ASSUMPTION_CHALLENGING
        elif "perspective" in problem_lower or "viewpoint" in problem_lower:
            return ThinkingPattern.PERSPECTIVE_SHIFTING  
        elif "combine" in problem_lower or "merge" in problem_lower:
            return ThinkingPattern.COMBINATION
        elif "opposite" in problem_lower or "reverse" in problem_lower:
            return ThinkingPattern.REVERSE
        elif "like" in problem_lower or "similar" in problem_lower:
            return ThinkingPattern.ANALOGICAL
        else:
            # Default to random stimulus for creative breakthrough
            return ThinkingPattern.RANDOM_STIMULUS
    
    async def _apply_thinking_pattern(self, problem: str, pattern: ThinkingPattern, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply specific lateral thinking pattern"""
        
        if pattern == ThinkingPattern.ANALOGICAL:
            return await self._analogical_thinking(problem, context)
        elif pattern == ThinkingPattern.REVERSE:
            return await self._reverse_thinking(problem, context)
        elif pattern == ThinkingPattern.RANDOM_STIMULUS:
            return await self._random_stimulus_thinking(problem, context)
        elif pattern == ThinkingPattern.ASSUMPTION_CHALLENGING:
            return await self._assumption_challenging(problem, context)
        elif pattern == ThinkingPattern.PERSPECTIVE_SHIFTING:
            return await self._perspective_shifting(problem, context)
        elif pattern == ThinkingPattern.COMBINATION:
            return await self._combination_thinking(problem, context)
        else:
            return await self._random_stimulus_thinking(problem, context)
    
    async def _analogical_thinking(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply analogical thinking pattern"""
        
        # Select random domain for analogy
        domain = random.choice(list(self.analogy_domains.keys()))
        concepts = self.analogy_domains[domain]
        selected_concept = random.choice(concepts)
        
        insight = f"Consider this problem like {selected_concept} in {domain}. "
        
        if domain == "nature":
            insight += f"How does {selected_concept} solve similar challenges? What adaptive strategies could apply here?"
        elif domain == "mechanics":
            insight += f"What mechanical principles of {selected_concept} could provide leverage for this problem?"
        elif domain == "architecture":
            insight += f"How would an architect approach this using principles of {selected_concept}?"
        elif domain == "music":
            insight += f"What musical concepts of {selected_concept} could create harmony in this situation?"
        elif domain == "cooking":
            insight += f"How would a chef apply {selected_concept} principles to blend solutions?"
        elif domain == "sports":
            insight += f"What strategic approaches from {selected_concept} could win this challenge?"
        elif domain == "art":
            insight += f"How would an artist use {selected_concept} to create something beautiful from this problem?"
        
        return {
            "insight": insight,
            "confidence": 0.7 + random.random() * 0.2,
            "novelty_score": 0.6 + random.random() * 0.3,
            "connections": [domain, selected_concept, "cross-domain thinking"],
            "evidence": [f"Analogical reasoning from {domain} domain", f"Pattern matching with {selected_concept}"],
            "applications": [f"Apply {selected_concept} principles", f"Explore {domain} strategies", "Cross-pollinate ideas"]
        }
    
    async def _reverse_thinking(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply reverse thinking pattern"""
        
        insight = f"Instead of solving '{problem}', consider: What if we did the complete opposite? "
        insight += "What would failure look like? What assumptions are we making? "
        insight += "Sometimes the solution emerges by reversing our approach entirely."
        
        return {
            "insight": insight,
            "confidence": 0.6 + random.random() * 0.3,
            "novelty_score": 0.7 + random.random() * 0.2,
            "connections": ["reverse logic", "assumption reversal", "opposite approach"],
            "evidence": ["Reverse thinking methodology", "Assumption challenging"],
            "applications": ["Identify hidden assumptions", "Consider opposite approaches", "Question the problem framing"]
        }
    
    async def _random_stimulus_thinking(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply random stimulus thinking pattern"""
        
        stimulus = random.choice(self.random_stimuli_database)
        
        insight = f"Let's connect '{problem}' with the random concept: '{stimulus}'. "
        insight += f"How might {stimulus} inspire a new approach? What qualities of {stimulus} "
        insight += f"could transform our understanding of this challenge? Sometimes breakthrough "
        insight += f"solutions come from the most unexpected connections."
        
        return {
            "insight": insight,
            "confidence": 0.5 + random.random() * 0.4,
            "novelty_score": 0.8 + random.random() * 0.2,
            "connections": [stimulus, "random association", "creative breakthrough"],
            "evidence": ["Random stimulus methodology", f"Connection with {stimulus}"],
            "applications": [f"Explore {stimulus} qualities", "Find unexpected connections", "Break mental patterns"]
        }
    
    async def _assumption_challenging(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply assumption challenging pattern"""
        
        insight = f"What assumptions are we making about '{problem}'? "
        insight += "What if the fundamental premise is wrong? What constraints are artificial? "
        insight += "Challenge every 'must be' and 'cannot be' - breakthrough solutions often "
        insight += "emerge when we question what everyone takes for granted."
        
        return {
            "insight": insight,
            "confidence": 0.8 + random.random() * 0.2,
            "novelty_score": 0.6 + random.random() * 0.3,
            "connections": ["assumption analysis", "constraint removal", "premise questioning"],
            "evidence": ["Assumption challenging methodology", "Constraint analysis"],
            "applications": ["List all assumptions", "Question constraints", "Reframe the problem"]
        }
    
    async def _perspective_shifting(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply perspective shifting pattern"""
        
        perspectives = ["child", "alien", "time traveler", "artist", "scientist", "philosopher"]
        perspective = random.choice(perspectives)
        
        insight = f"How would a {perspective} view '{problem}'? "
        insight += f"What would they see that we miss? What solutions would seem obvious to them? "
        insight += f"Shifting perspective often reveals solutions hiding in plain sight."
        
        return {
            "insight": insight,
            "confidence": 0.7 + random.random() * 0.2,
            "novelty_score": 0.7 + random.random() * 0.2,
            "connections": [perspective, "viewpoint shift", "cognitive flexibility"],
            "evidence": [f"Perspective shift to {perspective}", "Multi-viewpoint analysis"],
            "applications": [f"Adopt {perspective} mindset", "Explore alternative viewpoints", "Question standard approaches"]
        }
    
    async def _combination_thinking(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply combination thinking pattern"""
        
        elements = ["technology", "art", "nature", "science", "emotion", "logic"]
        element1, element2 = random.sample(elements, 2)
        
        insight = f"What if we combined {element1} and {element2} to approach '{problem}'? "
        insight += f"How might the fusion of {element1}'s strengths with {element2}'s "
        insight += f"capabilities create a novel solution? Innovation often happens at the intersection."
        
        return {
            "insight": insight,
            "confidence": 0.6 + random.random() * 0.3,
            "novelty_score": 0.8 + random.random() * 0.1,
            "connections": [element1, element2, "synthesis", "intersection thinking"],
            "evidence": [f"Combination of {element1} and {element2}", "Intersection analysis"],
            "applications": [f"Merge {element1} approaches", f"Integrate {element2} methods", "Find synthesis opportunities"]
        }
    
    async def generate_creative_solution(self, problem: str, constraints: List[str] = None, 
                                       inspiration_sources: List[str] = None) -> CreativeSolution:
        """Generate comprehensive creative solution using multiple lateral thinking approaches"""
        
        # Generate multiple insights using different patterns
        insights = []
        for pattern in [ThinkingPattern.ANALOGICAL, ThinkingPattern.RANDOM_STIMULUS, 
                       ThinkingPattern.ASSUMPTION_CHALLENGING]:
            insight = await self.generate_lateral_insight(problem)
            insights.append(insight)
        
        # Synthesize insights into creative solution
        solution_text = "Creative Multi-Pattern Solution:\n\n"
        steps_taken = []
        all_connections = []
        
        for i, insight in enumerate(insights, 1):
            solution_text += f"{i}. {insight.thinking_pattern.value.replace('_', ' ').title()} Approach:\n"
            solution_text += f"   {insight.insight}\n\n"
            steps_taken.append(f"Applied {insight.thinking_pattern.value} thinking")
            all_connections.extend(insight.connections_made)
        
        solution_text += "Synthesis: By combining these diverse perspectives, we can approach "
        solution_text += "the problem with unprecedented creativity and find breakthrough solutions "
        solution_text += "that conventional thinking might miss."
        
        solution = CreativeSolution(
            id=str(uuid.uuid4()),
            original_problem=problem,
            solution=solution_text,
            approach_used="Multi-pattern lateral thinking synthesis",
            creativity_score=0.8 + random.random() * 0.2,
            feasibility_score=0.6 + random.random() * 0.3,
            uniqueness_score=0.9 + random.random() * 0.1,
            steps_taken=steps_taken,
            inspiration_sources=inspiration_sources or ["lateral_thinking_patterns", "creative_synthesis"],
            timestamp=datetime.utcnow()
        )
        
        self.solutions_history.append(solution)
        return solution
    
    async def get_thinking_analytics(self) -> Dict[str, Any]:
        """Get analytics about lateral thinking patterns and performance"""
        
        total_insights = len(self.insights_history)
        total_solutions = len(self.solutions_history)
        
        if total_insights == 0:
            return {
                "total_insights": 0,
                "total_solutions": 0,
                "most_used_pattern": None,
                "average_confidence": 0,
                "average_novelty": 0,
                "creativity_evolution": []
            }
        
        avg_confidence = sum(insight.confidence for insight in self.insights_history) / total_insights
        avg_novelty = sum(insight.novelty_score for insight in self.insights_history) / total_insights
        
        most_used_pattern = max(self.thinking_patterns_used.items(), key=lambda x: x[1])[0] if self.thinking_patterns_used else None
        
        # Calculate creativity evolution over time
        creativity_evolution = []
        if total_solutions > 0:
            for solution in self.solutions_history[-10:]:  # Last 10 solutions
                creativity_evolution.append({
                    "timestamp": solution.timestamp.isoformat(),
                    "creativity_score": solution.creativity_score,
                    "uniqueness_score": solution.uniqueness_score
                })
        
        return {
            "total_insights": total_insights,
            "total_solutions": total_solutions,
            "most_used_pattern": most_used_pattern,
            "average_confidence": round(avg_confidence, 3),
            "average_novelty": round(avg_novelty, 3),
            "pattern_usage": self.thinking_patterns_used,
            "creativity_evolution": creativity_evolution,
            "recent_insights": [insight.to_dict() for insight in self.insights_history[-5:]]
        }
    
    async def get_lateral_insights_by_pattern(self, pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get insights filtered by thinking pattern"""
        
        filtered_insights = [
            insight for insight in self.insights_history 
            if insight.thinking_pattern.value == pattern
        ]
        
        return [insight.to_dict() for insight in filtered_insights[-limit:]]
    
    async def challenge_conventional_thinking(self, topic: str) -> Dict[str, Any]:
        """Challenge conventional thinking about a topic"""
        
        insight = await self.generate_lateral_insight(
            f"Challenge conventional thinking about: {topic}",
            {"focus": "assumption_challenging"}
        )
        
        return {
            "topic": topic,
            "conventional_challenges": insight.insight,
            "new_perspectives": insight.connections_made,
            "confidence": insight.confidence,
            "evidence": insight.evidence,
            "action_items": insight.applications
        }