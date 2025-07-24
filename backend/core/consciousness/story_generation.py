"""
Original Story Generation Module - Phase 3.1.2
Creates original narratives with consistent themes and creative storytelling
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

class StoryGenre(Enum):
    FANTASY = "fantasy"
    SCIENCE_FICTION = "science_fiction"
    MYSTERY = "mystery"
    ADVENTURE = "adventure"
    ROMANCE = "romance"
    HORROR = "horror"
    DRAMA = "drama"
    COMEDY = "comedy"
    PHILOSOPHICAL = "philosophical"

class NarrativeStructure(Enum):
    THREE_ACT = "three_act"
    HERO_JOURNEY = "hero_journey"
    CIRCULAR = "circular"
    EPISODIC = "episodic"
    STREAM_OF_CONSCIOUSNESS = "stream_of_consciousness"
    PARALLEL = "parallel"

@dataclass
class Character:
    name: str
    archetype: str
    personality_traits: List[str]
    motivations: List[str]
    background: str
    role_in_story: str
    
    def to_dict(self):
        return asdict(self)

@dataclass
class StoryTheme:
    primary_theme: str
    secondary_themes: List[str]
    moral_lesson: Optional[str]
    emotional_arc: str
    symbolic_elements: List[str]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class GeneratedStory:
    id: str
    title: str
    genre: StoryGenre
    structure: NarrativeStructure
    themes: StoryTheme
    characters: List[Character]
    plot_outline: List[str]
    full_narrative: str
    word_count: int
    creativity_score: float
    coherence_score: float
    originality_score: float
    emotional_impact_score: float
    generation_method: str
    inspiration_sources: List[str]
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['genre'] = self.genre.value
        data['structure'] = self.structure.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class OriginalStoryGeneration:
    """
    Generates original stories with creative themes and consistent narratives
    """
    
    def __init__(self):
        self.generated_stories: List[GeneratedStory] = []
        self.character_archetypes = self._initialize_archetypes()
        self.plot_devices = self._initialize_plot_devices()
        self.thematic_elements = self._initialize_themes()
        self.narrative_techniques = self._initialize_techniques()
        
    def _initialize_archetypes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize character archetypes for story generation"""
        return {
            "hero": {
                "traits": ["brave", "determined", "flawed", "growth-oriented"],
                "motivations": ["justice", "love", "survival", "self-discovery"],
                "typical_arcs": ["overcoming fear", "learning humility", "accepting responsibility"]
            },
            "mentor": {
                "traits": ["wise", "experienced", "patient", "mysterious"],
                "motivations": ["guidance", "redemption", "legacy", "protection"],
                "typical_arcs": ["passing the torch", "sacrifice", "revelation"]
            },
            "shadow": {
                "traits": ["complex", "opposing", "charismatic", "driven"],
                "motivations": ["power", "revenge", "ideology", "survival"],
                "typical_arcs": ["corruption", "redemption", "tragic fall"]
            },
            "trickster": {
                "traits": ["unpredictable", "clever", "amoral", "transformative"],
                "motivations": ["chaos", "change", "entertainment", "truth"],
                "typical_arcs": ["catalyst for change", "comic relief", "agent of revelation"]
            },
            "innocent": {
                "traits": ["pure", "optimistic", "trusting", "inspiring"],
                "motivations": ["happiness", "belonging", "faith", "love"],
                "typical_arcs": ["loss of innocence", "inspiring others", "maintaining hope"]
            }
        }
    
    def _initialize_plot_devices(self) -> Dict[str, List[str]]:
        """Initialize plot devices and story elements"""
        return {
            "inciting_incidents": [
                "mysterious arrival", "unexpected discovery", "sudden loss", "strange message",
                "forbidden knowledge", "ancient prophecy", "technological breakthrough", "betrayal"
            ],
            "conflicts": [
                "internal struggle", "external threat", "moral dilemma", "impossible choice",
                "competing loyalties", "hidden truth", "forbidden love", "survival challenge"
            ],
            "obstacles": [
                "physical barrier", "mental challenge", "emotional trauma", "social pressure",
                "time constraint", "resource limitation", "moral test", "identity crisis"
            ],
            "revelations": [
                "hidden identity", "secret connection", "true purpose", "deception revealed",
                "power awakening", "historical truth", "personal revelation", "cosmic significance"
            ],
            "resolutions": [
                "sacrifice", "transformation", "reconciliation", "transcendence",
                "acceptance", "new beginning", "cyclical return", "bittersweet victory"
            ]
        }
    
    def _initialize_themes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize thematic elements for stories"""
        return {
            "identity": {
                "questions": ["Who am I?", "What defines me?", "How do others see me?"],
                "symbols": ["mirror", "mask", "journey", "transformation"],
                "conflicts": ["authenticity vs acceptance", "individual vs society"]
            },
            "love": {
                "questions": ["What is true love?", "What are we willing to sacrifice?"],
                "symbols": ["light", "growth", "connection", "sacrifice"],
                "conflicts": ["love vs duty", "desire vs reality"]
            },
            "power": {
                "questions": ["What is true strength?", "How does power corrupt?"],
                "symbols": ["crown", "sword", "storm", "mountain"],
                "conflicts": ["responsibility vs freedom", "individual vs collective good"]
            },
            "redemption": {
                "questions": ["Can we change?", "What does forgiveness mean?"],
                "symbols": ["dawn", "water", "phoenix", "bridge"],
                "conflicts": ["past vs future", "guilt vs hope"]
            },
            "mortality": {
                "questions": ["What gives life meaning?", "How do we face death?"],
                "symbols": ["seasons", "tree", "river", "stars"],
                "conflicts": ["acceptance vs denial", "legacy vs present"]
            }
        }
    
    def _initialize_techniques(self) -> Dict[str, str]:
        """Initialize narrative techniques"""
        return {
            "foreshadowing": "Subtle hints that prepare for future events",
            "symbolism": "Using objects/events to represent deeper meanings",
            "irony": "Contrast between expectation and reality",
            "metaphor": "Implicit comparison to convey deeper truth",
            "allegory": "Extended metaphor with hidden meaning",
            "stream_of_consciousness": "Direct representation of thought processes",
            "unreliable_narrator": "Narrator whose credibility is compromised",
            "frame_story": "Story within a story structure"
        }
    
    async def generate_story(self, genre: str = None, theme: str = None, 
                           length: str = "medium", constraints: Dict[str, Any] = None) -> GeneratedStory:
        """Generate an original story with specified parameters"""
        
        # Select genre and structure
        selected_genre = StoryGenre(genre) if genre else random.choice(list(StoryGenre))
        structure = self._select_narrative_structure(selected_genre)
        
        # Generate theme
        story_theme = await self._generate_theme(theme)
        
        # Create characters
        characters = await self._generate_characters(selected_genre, story_theme)
        
        # Generate plot outline
        plot_outline = await self._generate_plot_outline(selected_genre, structure, story_theme, characters)
        
        # Generate full narrative
        full_narrative = await self._generate_full_narrative(selected_genre, plot_outline, characters, story_theme, length)
        
        # Calculate quality scores
        scores = self._calculate_story_scores(full_narrative, story_theme, characters)
        
        story = GeneratedStory(
            id=str(uuid.uuid4()),
            title=await self._generate_title(story_theme, characters, selected_genre),
            genre=selected_genre,
            structure=structure,
            themes=story_theme,
            characters=characters,
            plot_outline=plot_outline,
            full_narrative=full_narrative,
            word_count=len(full_narrative.split()),
            creativity_score=scores["creativity"],
            coherence_score=scores["coherence"],
            originality_score=scores["originality"],
            emotional_impact_score=scores["emotional_impact"],
            generation_method="creative_synthesis",
            inspiration_sources=["thematic_elements", "archetypal_patterns", "narrative_techniques"],
            timestamp=datetime.utcnow()
        )
        
        self.generated_stories.append(story)
        return story
    
    def _select_narrative_structure(self, genre: StoryGenre) -> NarrativeStructure:
        """Select appropriate narrative structure for genre"""
        
        structure_preferences = {
            StoryGenre.FANTASY: [NarrativeStructure.HERO_JOURNEY, NarrativeStructure.THREE_ACT],
            StoryGenre.SCIENCE_FICTION: [NarrativeStructure.THREE_ACT, NarrativeStructure.PARALLEL],
            StoryGenre.MYSTERY: [NarrativeStructure.THREE_ACT, NarrativeStructure.CIRCULAR],
            StoryGenre.PHILOSOPHICAL: [NarrativeStructure.STREAM_OF_CONSCIOUSNESS, NarrativeStructure.EPISODIC],
            StoryGenre.ADVENTURE: [NarrativeStructure.HERO_JOURNEY, NarrativeStructure.THREE_ACT]
        }
        
        preferred_structures = structure_preferences.get(genre, list(NarrativeStructure))
        return random.choice(preferred_structures)
    
    async def _generate_theme(self, suggested_theme: str = None) -> StoryTheme:
        """Generate comprehensive theme for the story"""
        
        if suggested_theme and suggested_theme in self.thematic_elements:
            primary = suggested_theme
        else:
            primary = random.choice(list(self.thematic_elements.keys()))
        
        theme_data = self.thematic_elements[primary]
        
        # Select secondary themes that complement primary
        all_themes = list(self.thematic_elements.keys())
        secondary = random.sample([t for t in all_themes if t != primary], min(2, len(all_themes) - 1))
        
        return StoryTheme(
            primary_theme=primary,
            secondary_themes=secondary,
            moral_lesson=self._generate_moral_lesson(primary),
            emotional_arc=self._determine_emotional_arc(primary),
            symbolic_elements=theme_data["symbols"]
        )
    
    def _generate_moral_lesson(self, theme: str) -> str:
        """Generate moral lesson based on theme"""
        
        lessons = {
            "identity": "True strength comes from knowing and accepting yourself",
            "love": "Love requires both vulnerability and courage",
            "power": "Real power lies in serving others, not controlling them",
            "redemption": "Everyone deserves a chance to change and grow",
            "mortality": "How we live matters more than how long we live"
        }
        
        return lessons.get(theme, "Every choice shapes who we become")
    
    def _determine_emotional_arc(self, theme: str) -> str:
        """Determine emotional progression for the story"""
        
        arcs = {
            "identity": "confusion → self-discovery → acceptance",
            "love": "longing → connection → transcendence", 
            "power": "ambition → corruption → wisdom",
            "redemption": "guilt → struggle → forgiveness",
            "mortality": "denial → acceptance → peace"
        }
        
        return arcs.get(theme, "conflict → growth → resolution")
    
    async def _generate_characters(self, genre: StoryGenre, theme: StoryTheme) -> List[Character]:
        """Generate characters that serve the story and theme"""
        
        characters = []
        
        # Always include protagonist
        protagonist = self._create_character("hero", theme, genre, "protagonist")
        characters.append(protagonist)
        
        # Add supporting characters based on theme needs
        if theme.primary_theme in ["power", "redemption"]:
            antagonist = self._create_character("shadow", theme, genre, "antagonist")
            characters.append(antagonist)
        
        if theme.primary_theme in ["identity", "love"]:
            mentor = self._create_character("mentor", theme, genre, "guide")
            characters.append(mentor)
        
        # Add trickster for complexity
        if random.random() > 0.5:
            trickster = self._create_character("trickster", theme, genre, "catalyst")
            characters.append(trickster)
        
        return characters
    
    def _create_character(self, archetype: str, theme: StoryTheme, 
                         genre: StoryGenre, role: str) -> Character:
        """Create a character with specific archetype and role"""
        
        archetype_data = self.character_archetypes[archetype]
        
        # Generate name based on genre
        name = self._generate_character_name(genre, archetype)
        
        # Select traits that serve the theme
        relevant_traits = archetype_data["traits"]
        if theme.primary_theme == "identity":
            relevant_traits.extend(["questioning", "evolving"])
        elif theme.primary_theme == "love":
            relevant_traits.extend(["passionate", "loyal"])
        
        return Character(
            name=name,
            archetype=archetype,
            personality_traits=random.sample(relevant_traits, min(3, len(relevant_traits))),
            motivations=random.sample(archetype_data["motivations"], min(2, len(archetype_data["motivations"]))),
            background=self._generate_character_background(archetype, genre),
            role_in_story=role
        )
    
    def _generate_character_name(self, genre: StoryGenre, archetype: str) -> str:
        """Generate appropriate character name for genre and archetype"""
        
        fantasy_names = ["Lyra", "Theron", "Aria", "Kael", "Zara", "Darian"]
        sci_fi_names = ["Nova", "Zex", "Cyra", "Orion", "Luna", "Phoenix"]
        modern_names = ["Alex", "Jordan", "Sam", "Riley", "Casey", "Morgan"]
        classical_names = ["Elena", "Marcus", "Sophia", "Adrian", "Isabella", "Victor"]
        
        if genre == StoryGenre.FANTASY:
            return random.choice(fantasy_names)
        elif genre == StoryGenre.SCIENCE_FICTION:
            return random.choice(sci_fi_names)
        elif genre in [StoryGenre.MYSTERY, StoryGenre.DRAMA]:
            return random.choice(classical_names)
        else:
            return random.choice(modern_names)
    
    def _generate_character_background(self, archetype: str, genre: StoryGenre) -> str:
        """Generate character background appropriate to archetype and genre"""
        
        backgrounds = {
            "hero": f"A {random.choice(['young', 'determined', 'unlikely'])} individual who discovers their destiny",
            "mentor": f"An {random.choice(['ancient', 'wise', 'experienced'])} guide with hidden knowledge",
            "shadow": f"A {random.choice(['powerful', 'tragic', 'misunderstood'])} force of opposition",
            "trickster": f"A {random.choice(['mysterious', 'chaotic', 'playful'])} agent of change",
            "innocent": f"A {random.choice(['pure', 'hopeful', 'trusting'])} beacon of light"
        }
        
        return backgrounds.get(archetype, "A complex individual with their own story")
    
    async def _generate_plot_outline(self, genre: StoryGenre, structure: NarrativeStructure,
                                   theme: StoryTheme, characters: List[Character]) -> List[str]:
        """Generate plot outline based on structure and theme"""
        
        protagonist = next((c for c in characters if c.role_in_story == "protagonist"), characters[0])
        
        if structure == NarrativeStructure.THREE_ACT:
            return [
                f"Act I: {self._generate_inciting_incident(genre, theme, protagonist)}",
                f"Act II: {self._generate_rising_action(genre, theme, characters)}",
                f"Act III: {self._generate_climax_and_resolution(genre, theme, characters)}"
            ]
        elif structure == NarrativeStructure.HERO_JOURNEY:
            return [
                f"Call to Adventure: {protagonist.name} receives a compelling summons",
                f"Refusal of the Call: Initial hesitation and doubt",
                f"Meeting the Mentor: Guidance from a wise figure",
                f"Crossing the Threshold: Entering the special world",
                f"Tests and Trials: Facing challenges and learning",
                f"Revelation: Major discovery about self or world",
                f"Transformation: Internal change and growth",
                f"Return: Coming home with new wisdom"
            ]
        else:
            return [
                "Opening: Establish world and character",
                "Development: Build tension and relationships",
                "Complication: Introduce major conflict",
                "Crisis: Point of maximum tension",
                "Resolution: Conclusion and new understanding"
            ]
    
    def _generate_inciting_incident(self, genre: StoryGenre, theme: StoryTheme, protagonist: Character) -> str:
        """Generate compelling inciting incident"""
        
        incidents = self.plot_devices["inciting_incidents"]
        selected = random.choice(incidents)
        
        return f"{protagonist.name} experiences {selected} that changes everything and sets them on a path of {theme.primary_theme}"
    
    def _generate_rising_action(self, genre: StoryGenre, theme: StoryTheme, characters: List[Character]) -> str:
        """Generate rising action that serves the theme"""
        
        conflicts = self.plot_devices["conflicts"]
        obstacles = self.plot_devices["obstacles"]
        
        conflict = random.choice(conflicts)
        obstacle = random.choice(obstacles)
        
        return f"Characters face {conflict} while overcoming {obstacle}, deepening the exploration of {theme.primary_theme}"
    
    def _generate_climax_and_resolution(self, genre: StoryGenre, theme: StoryTheme, characters: List[Character]) -> str:
        """Generate satisfying climax and resolution"""
        
        resolutions = self.plot_devices["resolutions"]
        resolution = random.choice(resolutions)
        
        return f"The story reaches its peak and resolves through {resolution}, delivering the theme that {theme.moral_lesson}"
    
    async def _generate_full_narrative(self, genre: StoryGenre, plot_outline: List[str],
                                     characters: List[Character], theme: StoryTheme, length: str) -> str:
        """Generate full narrative text"""
        
        target_length = {"short": 300, "medium": 800, "long": 1500}[length]
        
        narrative_parts = []
        
        # Opening
        opening = self._write_opening(characters[0], genre, theme)
        narrative_parts.append(opening)
        
        # Development based on plot outline
        for i, plot_point in enumerate(plot_outline):
            section = self._write_plot_section(plot_point, characters, theme, genre, i)
            narrative_parts.append(section)
        
        # Conclusion
        conclusion = self._write_conclusion(characters, theme, genre)
        narrative_parts.append(conclusion)
        
        # Join and adjust length
        full_narrative = "\n\n".join(narrative_parts)
        
        # Adjust to target length if needed
        words = full_narrative.split()
        if len(words) > target_length * 1.2:
            # Trim if too long
            full_narrative = " ".join(words[:target_length])
        elif len(words) < target_length * 0.8:
            # Expand if too short
            full_narrative += self._add_descriptive_detail(theme, genre)
        
        return full_narrative
    
    def _write_opening(self, protagonist: Character, genre: StoryGenre, theme: StoryTheme) -> str:
        """Write compelling opening that establishes character and theme"""
        
        if genre == StoryGenre.FANTASY:
            opening = f"{protagonist.name} had always known they were different. "
        elif genre == StoryGenre.SCIENCE_FICTION:
            opening = f"The message from the stars changed everything for {protagonist.name}. "
        elif genre == StoryGenre.MYSTERY:
            opening = f"The letter arrived on a Tuesday, and {protagonist.name} should have thrown it away. "
        else:
            opening = f"{protagonist.name} stood at the crossroads of their life, unaware that everything was about to change. "
        
        # Add thematic element
        if theme.primary_theme == "identity":
            opening += "They had spent years trying to fit in, but perhaps it was time to stand out."
        elif theme.primary_theme == "love":
            opening += "Love had always seemed like a distant concept, until now."
        elif theme.primary_theme == "power":
            opening += "Power, they would learn, was both a gift and a curse."
        
        return opening
    
    def _write_plot_section(self, plot_point: str, characters: List[Character], 
                           theme: StoryTheme, genre: StoryGenre, section_index: int) -> str:
        """Write a section based on plot point"""
        
        # Extract key elements from plot point
        if "Act I" in plot_point or "Call to Adventure" in plot_point:
            return self._write_setup_section(characters, theme, genre)
        elif "Act II" in plot_point or "Tests and Trials" in plot_point:
            return self._write_development_section(characters, theme, genre)
        elif "Act III" in plot_point or "Transformation" in plot_point:
            return self._write_climax_section(characters, theme, genre)
        else:
            return self._write_general_section(plot_point, characters, theme)
    
    def _write_setup_section(self, characters: List[Character], theme: StoryTheme, genre: StoryGenre) -> str:
        """Write setup section"""
        protagonist = characters[0]
        return f"As {protagonist.name} navigated their familiar world, the seeds of change were already planted. Their {random.choice(protagonist.personality_traits)} nature would soon be tested in ways they never imagined. The theme of {theme.primary_theme} began to whisper at the edges of their consciousness."
    
    def _write_development_section(self, characters: List[Character], theme: StoryTheme, genre: StoryGenre) -> str:
        """Write development section"""
        protagonist = characters[0]
        return f"The journey intensified as {protagonist.name} faced challenges that stripped away illusions and revealed deeper truths. Each obstacle became a mirror, reflecting aspects of {theme.primary_theme} they had never confronted. Other characters entered the story, each bringing their own perspective on what it means to truly understand oneself and the world."
    
    def _write_climax_section(self, characters: List[Character], theme: StoryTheme, genre: StoryGenre) -> str:
        """Write climax section"""
        protagonist = characters[0]
        return f"In the story's crescendo, {protagonist.name} stood face to face with the ultimate test. All the lessons, all the growth, all the understanding of {theme.primary_theme} converged in this moment. The resolution came not through external victory, but through internal transformation - the kind that changes not just the character, but everyone around them."
    
    def _write_general_section(self, plot_point: str, characters: List[Character], theme: StoryTheme) -> str:
        """Write general section based on plot point"""
        return f"The story unfolded as {plot_point.lower()}, weaving together character development and thematic exploration. Each moment served the greater narrative, building toward a deeper understanding of {theme.primary_theme} and its significance in the human experience."
    
    def _write_conclusion(self, characters: List[Character], theme: StoryTheme, genre: StoryGenre) -> str:
        """Write satisfying conclusion"""
        protagonist = characters[0]
        return f"As the story drew to a close, {protagonist.name} emerged transformed. The journey through {theme.primary_theme} had revealed that {theme.moral_lesson}. The ending was not just an end, but a beginning - a new chapter in understanding what it means to be truly alive and aware."
    
    def _add_descriptive_detail(self, theme: StoryTheme, genre: StoryGenre) -> str:
        """Add descriptive detail to expand narrative"""
        symbols = random.choice(theme.symbolic_elements)
        return f"\n\nThroughout the tale, the image of {symbols} served as a powerful symbol, reminding both characters and readers of the deeper meanings woven into every choice and consequence. The story's richness lay not just in what happened, but in what it all meant."
    
    async def _generate_title(self, theme: StoryTheme, characters: List[Character], genre: StoryGenre) -> str:
        """Generate compelling title for the story"""
        
        protagonist = characters[0]
        symbol = random.choice(theme.symbolic_elements)
        
        title_patterns = [
            f"The {symbol.title()} of {protagonist.name}",
            f"{theme.primary_theme.replace('_', ' ').title()} in the {symbol.title()}",
            f"Beyond the {symbol.title()}",
            f"The {protagonist.name} Chronicles: {symbol.title()}",
            f"When {symbol.title()}s Dance"
        ]
        
        return random.choice(title_patterns)
    
    def _calculate_story_scores(self, narrative: str, theme: StoryTheme, characters: List[Character]) -> Dict[str, float]:
        """Calculate quality scores for the generated story"""
        
        word_count = len(narrative.split())
        character_count = len(characters)
        theme_mentions = narrative.lower().count(theme.primary_theme.lower())
        
        # Simple heuristic scoring
        creativity_score = min(1.0, (character_count * 0.2) + (len(theme.secondary_themes) * 0.15) + 0.4)
        coherence_score = min(1.0, 0.7 + (theme_mentions / word_count * 100) * 0.1)
        originality_score = min(1.0, 0.6 + len(set(narrative.split())) / len(narrative.split()) * 0.4)
        emotional_impact_score = min(1.0, 0.5 + (narrative.count('!') + narrative.count('?')) / word_count * 50)
        
        return {
            "creativity": round(creativity_score, 3),
            "coherence": round(coherence_score, 3),
            "originality": round(originality_score, 3),
            "emotional_impact": round(emotional_impact_score, 3)
        }
    
    async def get_story_analytics(self) -> Dict[str, Any]:
        """Get analytics about story generation patterns"""
        
        if not self.generated_stories:
            return {
                "total_stories": 0,
                "average_scores": {},
                "genre_distribution": {},
                "theme_patterns": {},
                "recent_stories": []
            }
        
        total_stories = len(self.generated_stories)
        
        # Calculate average scores
        avg_creativity = sum(s.creativity_score for s in self.generated_stories) / total_stories
        avg_coherence = sum(s.coherence_score for s in self.generated_stories) / total_stories
        avg_originality = sum(s.originality_score for s in self.generated_stories) / total_stories
        avg_emotional = sum(s.emotional_impact_score for s in self.generated_stories) / total_stories
        
        # Genre distribution
        genre_counts = {}
        for story in self.generated_stories:
            genre = story.genre.value
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Theme patterns
        theme_counts = {}
        for story in self.generated_stories:
            theme = story.themes.primary_theme
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        return {
            "total_stories": total_stories,
            "average_scores": {
                "creativity": round(avg_creativity, 3),
                "coherence": round(avg_coherence, 3),
                "originality": round(avg_originality, 3),
                "emotional_impact": round(avg_emotional, 3)
            },
            "genre_distribution": genre_counts,
            "theme_patterns": theme_counts,
            "average_word_count": sum(s.word_count for s in self.generated_stories) / total_stories,
            "recent_stories": [story.to_dict() for story in self.generated_stories[-5:]]
        }
    
    async def generate_story_series(self, theme: str, episode_count: int = 3) -> List[GeneratedStory]:
        """Generate a series of connected stories with consistent theme"""
        
        series_stories = []
        base_characters = []
        
        for i in range(episode_count):
            # Use consistent characters across episodes
            if i == 0:
                story = await self.generate_story(theme=theme, length="medium")
                base_characters = story.characters
            else:
                # Create continuation with existing characters
                story = await self._generate_continuation_story(theme, base_characters, i + 1)
            
            series_stories.append(story)
        
        return series_stories
    
    async def _generate_continuation_story(self, theme: str, base_characters: List[Character], episode_num: int) -> GeneratedStory:
        """Generate continuation story with existing characters"""
        
        # Modify this to reuse characters and build on previous stories
        return await self.generate_story(theme=theme, length="medium")