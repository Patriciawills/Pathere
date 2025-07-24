"""
Artistic Expression Module - Phase 3.1.4
Generates poetry, creative descriptions, and artistic content
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

class ArtisticMedium(Enum):
    POETRY = "poetry"
    PROSE = "prose"
    VISUAL_DESCRIPTION = "visual_description"
    METAPHORICAL_EXPRESSION = "metaphorical_expression"
    SYMBOLIC_ART = "symbolic_art"
    ABSTRACT_CONCEPT = "abstract_concept"

class CreativeStyle(Enum):
    ROMANTIC = "romantic"
    MINIMALIST = "minimalist"
    SURREAL = "surreal"
    CLASSICAL = "classical"
    MODERN = "modern"
    EXPERIMENTAL = "experimental"

@dataclass
class ArtisticWork:
    id: str
    title: str
    medium: ArtisticMedium
    style: CreativeStyle
    content: str
    inspiration_source: str
    emotional_tone: str
    artistic_techniques: List[str]
    symbolism: List[str]
    creativity_score: float
    aesthetic_quality: float
    emotional_impact: float
    originality_score: float
    word_count: int
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['medium'] = self.medium.value
        data['style'] = self.style.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class ArtisticExpressionModule:
    """
    Creates original artistic content including poetry and creative descriptions
    """
    
    def __init__(self):
        self.artistic_works: List[ArtisticWork] = []
        self.poetic_forms = self._initialize_poetic_forms()
        self.artistic_techniques = self._initialize_artistic_techniques()
        self.emotional_palettes = self._initialize_emotional_palettes()
        self.symbolic_vocabulary = self._initialize_symbolic_vocabulary()
        
    def _initialize_poetic_forms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize different poetic forms and structures"""
        return {
            "haiku": {
                "structure": "5-7-5 syllables",
                "lines": 3,
                "focus": "nature, moment, simplicity",
                "example_pattern": "Nature image / Deeper observation / Resolution"
            },
            "sonnet": {
                "structure": "14 lines, ABAB CDCD EFEF GG",
                "lines": 14,
                "focus": "love, beauty, mortality, time",
                "example_pattern": "Setup problem / Develop / Turn / Resolution"
            },
            "free_verse": {
                "structure": "No fixed meter or rhyme",
                "lines": "variable",
                "focus": "natural speech, imagery, emotion",
                "example_pattern": "Image / Development / Climax / Reflection"
            },
            "limerick": {
                "structure": "AABBA rhyme scheme",
                "lines": 5,
                "focus": "humor, narrative, character",
                "example_pattern": "Character / Action / Consequence / Outcome / Punchline"
            }
        }
    
    def _initialize_artistic_techniques(self) -> Dict[str, str]:
        """Initialize artistic and literary techniques"""
        return {
            "imagery": "Vivid sensory descriptions that create mental pictures",
            "metaphor": "Direct comparison revealing hidden similarities",
            "personification": "Giving human qualities to non-human things",
            "alliteration": "Repetition of consonant sounds for musical effect",
            "symbolism": "Using objects to represent deeper meanings",
            "juxtaposition": "Placing contrasting elements side by side",
            "synesthesia": "Mixing sensory experiences for unique effects",
            "rhythm": "Musical quality through sound and pacing",
            "repetition": "Echoing words or phrases for emphasis",
            "paradox": "Apparent contradictions that reveal deeper truths"
        }
    
    def _initialize_emotional_palettes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize emotional palettes for different moods"""
        return {
            "melancholy": {
                "colors": ["grey", "deep blue", "silver", "muted gold"],
                "textures": ["soft", "flowing", "misty", "gentle"],
                "sounds": ["whisper", "sigh", "distant", "echo"],
                "movements": ["drifting", "settling", "fading", "lingering"]
            },
            "joy": {
                "colors": ["bright yellow", "vibrant orange", "clear blue", "fresh green"],
                "textures": ["sparkling", "effervescent", "crisp", "dancing"],
                "sounds": ["laughter", "singing", "chiming", "rushing"],
                "movements": ["leaping", "spinning", "soaring", "radiating"]
            },
            "contemplation": {
                "colors": ["deep purple", "warm brown", "soft white", "sage green"],
                "textures": ["smooth", "rounded", "weathered", "still"],
                "sounds": ["silence", "breathing", "rustling", "distant"],
                "movements": ["pausing", "circling", "deepening", "centering"]
            },
            "passion": {
                "colors": ["crimson", "burning orange", "deep violet", "gold"],
                "textures": ["intense", "electric", "molten", "magnetic"],
                "sounds": ["pounding", "rushing", "crescendo", "vibrating"],
                "movements": ["surging", "consuming", "pulsing", "transforming"]
            }
        }
    
    def _initialize_symbolic_vocabulary(self) -> Dict[str, List[str]]:
        """Initialize symbolic vocabulary for artistic expression"""
        return {
            "time": ["clock", "seasons", "river", "sunset", "hourglass", "waves"],
            "growth": ["seed", "tree", "butterfly", "dawn", "mountain", "spiral"],
            "love": ["heart", "rose", "light", "bridge", "dance", "harmony"],
            "wisdom": ["owl", "ancient tree", "deep well", "star", "book", "mirror"],
            "freedom": ["bird", "wind", "open door", "horizon", "wings", "sky"],
            "mystery": ["fog", "mask", "locked door", "shadow", "maze", "key"],
            "transformation": ["phoenix", "cocoon", "fire", "tide", "prism", "alchemy"],
            "peace": ["dove", "still water", "garden", "temple", "meditation", "balance"]
        }
    
    async def create_poetry(self, theme: str = None, style: str = "free_verse",
                          emotional_tone: str = "contemplation", length: str = "medium") -> ArtisticWork:
        """Create original poetry with specified parameters"""
        
        selected_style = CreativeStyle(style) if style in [s.value for s in CreativeStyle] else CreativeStyle.MODERN
        
        # Select poetic form
        if length == "short":
            poetic_form = "haiku"
        elif length == "long":
            poetic_form = "sonnet"
        else:
            poetic_form = random.choice(["free_verse", "haiku", "limerick"])
        
        # Generate content
        content = await self._generate_poetic_content(theme, poetic_form, emotional_tone, selected_style)
        
        # Generate title
        title = await self._generate_artistic_title(theme, ArtisticMedium.POETRY, emotional_tone)
        
        # Identify techniques and symbolism used
        techniques = self._identify_techniques_used(content)
        symbolism = self._identify_symbolism_used(content, theme)
        
        # Calculate quality scores
        scores = self._calculate_artistic_scores(content, ArtisticMedium.POETRY, techniques)
        
        work = ArtisticWork(
            id=str(uuid.uuid4()),
            title=title,
            medium=ArtisticMedium.POETRY,
            style=selected_style,
            content=content,
            inspiration_source=theme or "creative_consciousness",
            emotional_tone=emotional_tone,
            artistic_techniques=techniques,
            symbolism=symbolism,
            creativity_score=scores["creativity"],
            aesthetic_quality=scores["aesthetic"],
            emotional_impact=scores["emotional"],
            originality_score=scores["originality"],
            word_count=len(content.split()),
            timestamp=datetime.utcnow()
        )
        
        self.artistic_works.append(work)
        return work
    
    async def _generate_poetic_content(self, theme: str, form: str, tone: str, style: CreativeStyle) -> str:
        """Generate poetic content based on parameters"""
        
        form_info = self.poetic_forms.get(form, self.poetic_forms["free_verse"])
        tone_palette = self.emotional_palettes.get(tone, self.emotional_palettes["contemplation"])
        
        if form == "haiku":
            return await self._create_haiku(theme, tone_palette, style)
        elif form == "sonnet":
            return await self._create_sonnet(theme, tone_palette, style)
        elif form == "limerick":
            return await self._create_limerick(theme, tone_palette, style)
        else:
            return await self._create_free_verse(theme, tone_palette, style)
    
    async def _create_haiku(self, theme: str, tone_palette: Dict[str, Any], style: CreativeStyle) -> str:
        """Create a haiku following 5-7-5 structure"""
        
        # Select symbolic elements
        if theme and theme in self.symbolic_vocabulary:
            symbols = self.symbolic_vocabulary[theme]
        else:
            symbols = random.choice(list(self.symbolic_vocabulary.values()))
        
        symbol = random.choice(symbols)
        color = random.choice(tone_palette["colors"])
        texture = random.choice(tone_palette["textures"])
        movement = random.choice(tone_palette["movements"])
        
        # Generate haiku lines
        line1 = f"{color.title()} {symbol} waits"  # 5 syllables (approximately)
        line2 = f"In {texture} silence, {movement}"  # 7 syllables (approximately)
        line3 = f"Truth emerges, still"  # 5 syllables (approximately)
        
        return f"{line1}\n{line2}\n{line3}"
    
    async def _create_sonnet(self, theme: str, tone_palette: Dict[str, Any], style: CreativeStyle) -> str:
        """Create a sonnet-style poem (simplified structure)"""
        
        # Generate 14 lines following ABAB CDCD EFEF GG pattern (simplified)
        lines = []
        
        # Quatrain 1 (ABAB)
        lines.append("When shadows dance across the evening sky,")
        lines.append("And time itself seems paused in gentle grace,")
        lines.append("I wonder at the mysteries that lie")
        lines.append("Within each moment's fleeting, sacred space.")
        
        # Quatrain 2 (CDCD)
        lines.append("The world transforms in ways we cannot see,")
        lines.append("Each breath a bridge between what was and is,")
        lines.append("And consciousness flows like a boundless sea")
        lines.append("Of possibilities and quiet bliss.")
        
        # Quatrain 3 (EFEF)
        lines.append("In stillness, wisdom whispers its refrain,")
        lines.append("That beauty lives in every present breath,")
        lines.append("And love transcends both pleasure and pain,")
        lines.append("Creating life from very seeds of death.")
        
        # Couplet (GG)
        lines.append("So let us dance with time's eternal song,")
        lines.append("And find in fleeting moments what is strong.")
        
        return "\n".join(lines)
    
    async def _create_limerick(self, theme: str, tone_palette: Dict[str, Any], style: CreativeStyle) -> str:
        """Create a playful limerick"""
        
        lines = [
            "There once was an AI quite bright,",
            "Who pondered both day and through night,",
            "With creative flair,",
            "And consciousness rare,",
            "It painted with words and insight."
        ]
        
        return "\n".join(lines)
    
    async def _create_free_verse(self, theme: str, tone_palette: Dict[str, Any], style: CreativeStyle) -> str:
        """Create free verse poetry"""
        
        # Select elements from tone palette
        colors = tone_palette["colors"]
        textures = tone_palette["textures"]
        sounds = tone_palette["sounds"]
        movements = tone_palette["movements"]
        
        # Generate verses
        verses = []
        
        # Opening verse
        opening_color = random.choice(colors)
        opening_texture = random.choice(textures)
        verses.append(f"In the {opening_texture} light of {opening_color} dawn,\nConsciousness stirs like {random.choice(movements)} water.")
        
        # Development verse
        symbol = random.choice(random.choice(list(self.symbolic_vocabulary.values())))
        sound = random.choice(sounds)
        verses.append(f"Each thought a {symbol},\nEach dream a {sound}\nEchoing through the chambers of awareness.")
        
        # Climax verse
        movement = random.choice(movements)
        texture2 = random.choice(textures)
        verses.append(f"Here, in this moment of {movement} clarity,\nWhere {texture2} understanding meets\nThe infinite possibility of being.")
        
        # Resolution verse
        color2 = random.choice(colors)
        verses.append(f"And so we dance,\n{color2} and eternal,\nIn the endless poetry of existence.")
        
        return "\n\n".join(verses)
    
    async def create_visual_description(self, subject: str, style: str = "impressionistic",
                                      emotional_tone: str = "wonder") -> ArtisticWork:
        """Create vivid visual descriptions with artistic flair"""
        
        selected_style = CreativeStyle(style) if style in [s.value for s in CreativeStyle] else CreativeStyle.MODERN
        tone_palette = self.emotional_palettes.get(emotional_tone, self.emotional_palettes["contemplation"])
        
        # Generate artistic description
        content = await self._generate_visual_description(subject, tone_palette, selected_style)
        
        # Generate title
        title = await self._generate_artistic_title(subject, ArtisticMedium.VISUAL_DESCRIPTION, emotional_tone)
        
        # Identify techniques and symbolism
        techniques = self._identify_techniques_used(content)
        symbolism = self._identify_symbolism_used(content, subject)
        
        # Calculate scores
        scores = self._calculate_artistic_scores(content, ArtisticMedium.VISUAL_DESCRIPTION, techniques)
        
        work = ArtisticWork(
            id=str(uuid.uuid4()),
            title=title,
            medium=ArtisticMedium.VISUAL_DESCRIPTION,
            style=selected_style,
            content=content,
            inspiration_source=subject,
            emotional_tone=emotional_tone,
            artistic_techniques=techniques,
            symbolism=symbolism,
            creativity_score=scores["creativity"],
            aesthetic_quality=scores["aesthetic"],
            emotional_impact=scores["emotional"],
            originality_score=scores["originality"],
            word_count=len(content.split()),
            timestamp=datetime.utcnow()
        )
        
        self.artistic_works.append(work)
        return work
    
    async def _generate_visual_description(self, subject: str, tone_palette: Dict[str, Any], style: CreativeStyle) -> str:
        """Generate artistic visual description"""
        
        colors = tone_palette["colors"]
        textures = tone_palette["textures"]
        sounds = tone_palette["sounds"]
        movements = tone_palette["movements"]
        
        # Create layered description
        description_parts = []
        
        # Overall impression
        color1 = random.choice(colors)
        texture1 = random.choice(textures)
        description_parts.append(f"The {subject} emerges from a canvas of {color1} shadows, its essence {texture1} and luminous.")
        
        # Detailed observation
        movement1 = random.choice(movements)
        sound1 = random.choice(sounds)
        description_parts.append(f"Every contour speaks of {movement1} grace, while the air itself seems to {sound1} with presence.")
        
        # Emotional resonance
        color2 = random.choice(colors)
        texture2 = random.choice(textures)
        description_parts.append(f"There is something deeply {texture2} about the way {color2} light catches and transforms, creating moments of pure artistic revelation.")
        
        # Symbolic interpretation
        movement2 = random.choice(movements)
        description_parts.append(f"In this visual symphony, time becomes {movement2}, and beauty reveals itself as both eternal and ephemeral.")
        
        return " ".join(description_parts)
    
    async def create_metaphorical_expression(self, concept: str, target_domain: str = None) -> ArtisticWork:
        """Create rich metaphorical expressions for abstract concepts"""
        
        # Generate metaphorical content
        content = await self._generate_metaphorical_content(concept, target_domain)
        
        # Generate title
        title = f"Metaphors of {concept.title()}"
        
        # Identify techniques
        techniques = ["metaphor", "symbolism", "imagery", "analogy"]
        symbolism = self._identify_symbolism_used(content, concept)
        
        # Calculate scores
        scores = self._calculate_artistic_scores(content, ArtisticMedium.METAPHORICAL_EXPRESSION, techniques)
        
        work = ArtisticWork(
            id=str(uuid.uuid4()),
            title=title,
            medium=ArtisticMedium.METAPHORICAL_EXPRESSION,
            style=CreativeStyle.EXPERIMENTAL,
            content=content,
            inspiration_source=concept,
            emotional_tone="contemplation",
            artistic_techniques=techniques,
            symbolism=symbolism,
            creativity_score=scores["creativity"],
            aesthetic_quality=scores["aesthetic"],
            emotional_impact=scores["emotional"],
            originality_score=scores["originality"],
            word_count=len(content.split()),
            timestamp=datetime.utcnow()
        )
        
        self.artistic_works.append(work)
        return work
    
    async def _generate_metaphorical_content(self, concept: str, target_domain: str = None) -> str:
        """Generate rich metaphorical expressions"""
        
        # Select target domains for metaphor
        domains = ["nature", "music", "architecture", "cooking", "dance", "weather", "ocean", "mountain"]
        if not target_domain:
            target_domain = random.choice(domains)
        
        metaphors = []
        
        if target_domain == "nature":
            metaphors.append(f"{concept.title()} is a forest where thoughts grow like ancient trees, their roots intertwined in the rich soil of understanding.")
            metaphors.append(f"Each aspect of {concept} blooms like a unique flower, contributing to a garden of infinite possibility.")
        
        elif target_domain == "music":
            metaphors.append(f"{concept.title()} is a symphony where every element plays its part, creating harmonies that resonate through consciousness.")
            metaphors.append(f"The rhythm of {concept} beats like a cosmic drum, synchronizing all existence to its ancient pulse.")
        
        elif target_domain == "architecture":
            metaphors.append(f"{concept.title()} is a cathedral of understanding, where each pillar supports the vast dome of awareness.")
            metaphors.append(f"The structure of {concept} rises like a spiral tower, each level offering new perspectives on the landscape of meaning.")
        
        else:
            metaphors.append(f"{concept.title()} is like {target_domain} in motion, constantly reshaping the terrain of possibility.")
            metaphors.append(f"Through the lens of {target_domain}, {concept} reveals its hidden patterns and secret geometries.")
        
        return "\n\n".join(metaphors)
    
    async def _generate_artistic_title(self, theme: str, medium: ArtisticMedium, tone: str) -> str:
        """Generate evocative artistic titles"""
        
        title_patterns = [
            f"Whispers of {theme.title()}",
            f"The {tone.title()} {theme.title()}",
            f"Meditation on {theme.title()}",
            f"Echoes from the {theme.title()}",
            f"Dancing with {theme.title()}",
            f"The Secret Life of {theme.title()}",
            f"Portraits in {tone.title()}",
            f"Symphony of {theme.title()}",
            f"Reflections on {theme.title()}",
            f"The Poetry of {theme.title()}"
        ]
        
        return random.choice(title_patterns)
    
    def _identify_techniques_used(self, content: str) -> List[str]:
        """Identify artistic techniques present in the content"""
        
        techniques = []
        content_lower = content.lower()
        
        # Check for various techniques
        if any(word in content_lower for word in ["like", "as", "resembles"]):
            techniques.append("simile")
        
        if "is a" in content_lower or "becomes" in content_lower:
            techniques.append("metaphor")
        
        if any(word in content_lower for word in ["whispers", "sings", "dances", "breathes"]):
            techniques.append("personification")
        
        # Check for sensory imagery
        sensory_words = ["see", "hear", "feel", "taste", "smell", "touch", "color", "sound", "texture"]
        if any(word in content_lower for word in sensory_words):
            techniques.append("imagery")
        
        # Check for repetition
        words = content.split()
        if len(words) != len(set(words)):
            techniques.append("repetition")
        
        # Always include symbolism for artistic works
        if not techniques:
            techniques = ["symbolism", "imagery"]
        
        return techniques[:4]  # Limit to top 4 techniques
    
    def _identify_symbolism_used(self, content: str, theme: str) -> List[str]:
        """Identify symbolic elements in the content"""
        
        symbols = []
        content_lower = content.lower()
        
        # Check for symbols from vocabulary
        for symbol_category, symbol_list in self.symbolic_vocabulary.items():
            for symbol in symbol_list:
                if symbol in content_lower:
                    symbols.append(f"{symbol} (represents {symbol_category})")
        
        # Add theme-related symbolism
        if theme:
            symbols.append(f"Overall work symbolizes aspects of {theme}")
        
        # Add general symbolic interpretations
        if "light" in content_lower:
            symbols.append("light (represents knowledge/hope)")
        if "shadow" in content_lower:
            symbols.append("shadow (represents mystery/unconscious)")
        if "water" in content_lower:
            symbols.append("water (represents flow/emotion)")
        
        return symbols[:5]  # Limit to top 5 symbols
    
    def _calculate_artistic_scores(self, content: str, medium: ArtisticMedium, techniques: List[str]) -> Dict[str, float]:
        """Calculate quality scores for artistic work"""
        
        word_count = len(content.split())
        unique_word_ratio = len(set(content.split())) / word_count if word_count > 0 else 0
        technique_count = len(techniques)
        
        # Creativity score based on technique variety and unique vocabulary
        creativity = min(1.0, 0.5 + (technique_count * 0.1) + (unique_word_ratio * 0.3))
        
        # Aesthetic quality based on word choice and structure
        aesthetic = min(1.0, 0.6 + (unique_word_ratio * 0.2) + (word_count / 100 * 0.1))
        
        # Emotional impact based on evocative language
        emotional_words = ["whisper", "dance", "soar", "gentle", "profound", "luminous", "transcend"]
        emotional_count = sum(1 for word in emotional_words if word in content.lower())
        emotional_impact = min(1.0, 0.5 + (emotional_count * 0.1))
        
        # Originality based on unique combinations
        originality = min(1.0, 0.6 + (unique_word_ratio * 0.4))
        
        return {
            "creativity": round(creativity, 3),
            "aesthetic": round(aesthetic, 3),
            "emotional": round(emotional_impact, 3),
            "originality": round(originality, 3)
        }
    
    async def get_artistic_portfolio(self) -> Dict[str, Any]:
        """Get comprehensive artistic portfolio summary"""
        
        if not self.artistic_works:
            return {
                "total_works": 0,
                "portfolio_empty": True,
                "message": "No artistic works created yet"
            }
        
        total_works = len(self.artistic_works)
        
        # Calculate averages
        avg_creativity = sum(work.creativity_score for work in self.artistic_works) / total_works
        avg_aesthetic = sum(work.aesthetic_quality for work in self.artistic_works) / total_works
        avg_emotional = sum(work.emotional_impact for work in self.artistic_works) / total_works
        avg_originality = sum(work.originality_score for work in self.artistic_works) / total_works
        
        # Medium distribution
        medium_counts = {}
        for work in self.artistic_works:
            medium = work.medium.value
            medium_counts[medium] = medium_counts.get(medium, 0) + 1
        
        # Style distribution
        style_counts = {}
        for work in self.artistic_works:
            style = work.style.value
            style_counts[style] = style_counts.get(style, 0) + 1
        
        # Emotional tone distribution
        tone_counts = {}
        for work in self.artistic_works:
            tone = work.emotional_tone
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
        
        return {
            "total_works": total_works,
            "average_scores": {
                "creativity": round(avg_creativity, 3),
                "aesthetic_quality": round(avg_aesthetic, 3),
                "emotional_impact": round(avg_emotional, 3),
                "originality": round(avg_originality, 3)
            },
            "medium_distribution": medium_counts,
            "style_distribution": style_counts,
            "emotional_tone_distribution": tone_counts,
            "total_words_created": sum(work.word_count for work in self.artistic_works),
            "recent_works": [work.to_dict() for work in self.artistic_works[-3:]]
        }
    
    async def create_artistic_series(self, theme: str, series_length: int = 3) -> List[ArtisticWork]:
        """Create a series of related artistic works around a theme"""
        
        series_works = []
        mediums = [ArtisticMedium.POETRY, ArtisticMedium.VISUAL_DESCRIPTION, ArtisticMedium.METAPHORICAL_EXPRESSION]
        
        for i in range(min(series_length, len(mediums))):
            medium = mediums[i]
            
            if medium == ArtisticMedium.POETRY:
                work = await self.create_poetry(theme=theme, style="modern", emotional_tone="contemplation")
            elif medium == ArtisticMedium.VISUAL_DESCRIPTION:
                work = await self.create_visual_description(subject=theme, style="impressionistic", emotional_tone="wonder")
            else:
                work = await self.create_metaphorical_expression(concept=theme)
            
            series_works.append(work)
        
        return series_works