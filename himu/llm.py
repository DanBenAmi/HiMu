"""Abstract LLM interface and implementations for query parsing."""

from abc import ABC, abstractmethod
from typing import Optional, List, Literal, Dict
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, will rely on system env vars


class LogicalTreeNode(BaseModel):
    """Recursive Pydantic model for the logical query tree."""

    op: Literal["AND", "OR", "SEQ", "RIGHT_AFTER", "LEAF"]
    expert: Optional[Literal["OCR", "OVD", "CLIP", "ASR", "CLAP"]] = None
    query: Optional[str] = None
    children: Optional[List["LogicalTreeNode"]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "op": "AND",
                "children": [
                    {"op": "LEAF", "expert": "OCR", "query": "Welcome"},
                    {"op": "LEAF", "expert": "OVD", "query": "red car"}
                ]
            }
        }

    def model_post_init(self, __context):
        """Validate tree structure."""
        if self.op == "LEAF":
            if self.expert is None or self.query is None:
                raise ValueError("LEAF nodes must have 'expert' and 'query' fields")
            if self.children is not None:
                raise ValueError("LEAF nodes cannot have children")
        else:
            if self.children is None or len(self.children) == 0:
                raise ValueError(f"{self.op} nodes must have children")
            if self.op == "SEQ" and len(self.children) < 2:
                raise ValueError("SEQ nodes must have at least 2 children in chronological order")
            if self.op == "RIGHT_AFTER" and len(self.children) != 2:
                raise ValueError("RIGHT_AFTER nodes must have exactly 2 children (cause, effect)")


import logging
_repair_log = logging.getLogger(__name__)


def repair_tree(node: dict) -> dict:
    """Auto-repair common structural issues in a raw tree dict before validation.

    Fixes:
    - RIGHT_AFTER with >2 children: chains into nested RIGHT_AFTER pairs (right-associative).
      e.g. RIGHT_AFTER(A, B, C) -> RIGHT_AFTER(A, RIGHT_AFTER(B, C))
    """
    if not isinstance(node, dict):
        return node

    # Recursively repair children first
    if "children" in node and node["children"] is not None:
        node["children"] = [repair_tree(child) for child in node["children"]]

    # Fix RIGHT_AFTER with >2 children
    if node.get("op") == "RIGHT_AFTER" and node.get("children") and len(node["children"]) > 2:
        children = node["children"]
        _repair_log.info(
            f"Repairing RIGHT_AFTER node with {len(children)} children -> nested RIGHT_AFTER pairs"
        )
        # Right-associative nesting: RIGHT_AFTER(A, B, C, D) -> RIGHT_AFTER(A, RIGHT_AFTER(B, RIGHT_AFTER(C, D)))
        result = children[-1]
        for i in range(len(children) - 2, 0, -1):
            result = {"op": "RIGHT_AFTER", "children": [children[i], result]}
        node["children"] = [children[0], result]

    return node


def _expert_literals_str(use_asr: bool, use_clap: bool) -> str:
    """Build the expert literal string for output format sections."""
    experts = ['"OCR"', '"OVD"', '"CLIP"']
    if use_asr:
        experts.append('"ASR"')
    if use_clap:
        experts.append('"CLAP"')
    return " | ".join(experts)


def _build_examples(use_asr: bool, use_clap: bool) -> str:
    """Build examples section based on available experts."""
    examples = ""

    if use_clap and use_asr:
        examples += """
Example 1 - Temporal trigger with audio event:
Q: "After the doorbell rings, who opens the door?" Options: ["A man", "A woman"]
{"op":"SEQ","children":[{"op":"LEAF","expert":"CLAP","query":"doorbell ringing"},{"op":"AND","children":[{"op":"LEAF","expert":"CLIP","query":"opening door"},{"op":"LEAF","expert":"ASR","query":"door"},{"op":"OR","children":[{"op":"LEAF","expert":"OVD","query":"man"},{"op":"LEAF","expert":"OVD","query":"woman"}]}]}]}

Example 2 - OCR + ASR for on-screen text with speech:
Q: "What safety text appears when the instructor says caution?" Options: ["Wear goggles", "Keep distance"]
{"op":"AND","children":[{"op":"LEAF","expert":"ASR","query":"caution"},{"op":"LEAF","expert":"OVD","query":"person"},{"op":"OR","children":[{"op":"AND","children":[{"op":"LEAF","expert":"OCR","query":"goggles"},{"op":"LEAF","expert":"OVD","query":"safety goggles"}]},{"op":"LEAF","expert":"OCR","query":"keep distance"}]}]}

Example 3 - Multimodal with CLAP for sounds:
Q: "What happens during the chemical reaction?" Options: ["It bubbles", "It changes color", "It explodes"]
{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"beaker"},{"op":"LEAF","expert":"ASR","query":"reaction"},{"op":"OR","children":[{"op":"AND","children":[{"op":"LEAF","expert":"CLIP","query":"liquid bubbling"},{"op":"LEAF","expert":"ASR","query":"bubbles"}]},{"op":"AND","children":[{"op":"LEAF","expert":"CLIP","query":"liquid changing color"},{"op":"LEAF","expert":"ASR","query":"color"}]},{"op":"AND","children":[{"op":"LEAF","expert":"CLIP","query":"explosion"},{"op":"LEAF","expert":"CLAP","query":"explosion sound"}]}]}]}

Example 4 - Ordering question (order unknown, use OR not SEQ):
Q: "In which order are these tools used?" Options: represent different orderings of (a) Brush (b) Knife (c) Tape
{"op":"AND","children":[{"op":"LEAF","expert":"ASR","query":"tools"},{"op":"OR","children":[{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"brush"},{"op":"LEAF","expert":"CLIP","query":"using brush"}]},{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"knife"},{"op":"LEAF","expert":"CLIP","query":"cutting"}]},{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"tape"},{"op":"LEAF","expert":"CLIP","query":"taping"}]}]}]}"""

    elif use_asr:
        examples += """
Example 1 - Speech-triggered with visual verification:
Q: "What does the presenter show right after saying 'let me demonstrate'?" Options: ["A chart", "A product"]
{"op":"RIGHT_AFTER","children":[{"op":"LEAF","expert":"ASR","query":"demonstrate"},{"op":"OR","children":[{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"chart"},{"op":"LEAF","expert":"CLIP","query":"showing chart"}]},{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"product"},{"op":"LEAF","expert":"CLIP","query":"showing product"}]}]}]}

Example 2 - Scientific video with ASR overlap:
Q: "What happens during the chemical reaction?" Options: ["It bubbles", "It changes color", "It explodes"]
{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"beaker"},{"op":"LEAF","expert":"ASR","query":"reaction"},{"op":"OR","children":[{"op":"AND","children":[{"op":"LEAF","expert":"CLIP","query":"liquid bubbling"},{"op":"LEAF","expert":"ASR","query":"bubbles"}]},{"op":"AND","children":[{"op":"LEAF","expert":"CLIP","query":"liquid changing color"},{"op":"LEAF","expert":"ASR","query":"color"}]},{"op":"LEAF","expert":"CLIP","query":"explosion"}]}]}

Example 3 - Ordering question (order unknown, use OR not SEQ):
Q: "In which order are these tools used?" Options: represent different orderings of (a) Brush (b) Knife (c) Tape
{"op":"AND","children":[{"op":"LEAF","expert":"ASR","query":"tools"},{"op":"OR","children":[{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"brush"},{"op":"LEAF","expert":"CLIP","query":"using brush"}]},{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"knife"},{"op":"LEAF","expert":"CLIP","query":"cutting"}]},{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"tape"},{"op":"LEAF","expert":"CLIP","query":"taping"}]}]}]}"""

    elif use_clap:
        examples += """
Example 1 - Audio-triggered temporal query:
Q: "After the doorbell rings, who opens the door?" Options: ["A man", "A woman"]
{"op":"SEQ","children":[{"op":"LEAF","expert":"CLAP","query":"doorbell ringing"},{"op":"AND","children":[{"op":"LEAF","expert":"CLIP","query":"opening door"},{"op":"OR","children":[{"op":"LEAF","expert":"OVD","query":"man"},{"op":"LEAF","expert":"OVD","query":"woman"}]}]}]}

Example 2 - Temporal sequence with visual context:
Q: "What does the man do after the car leaves?" Options: ["Running", "Sleeping"]
{"op":"SEQ","children":[{"op":"LEAF","expert":"CLIP","query":"car leaving"},{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"person"},{"op":"OR","children":[{"op":"LEAF","expert":"CLIP","query":"running"},{"op":"LEAF","expert":"CLIP","query":"sleeping"}]}]}]}"""

    else:
        examples += """
Example 1 - Question with Options:
Q: "What is the chef wearing while cooking?" Options: ["Blue apron", "Red jacket", "Green hat"]
{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"chef"},{"op":"LEAF","expert":"CLIP","query":"person cooking in kitchen"},{"op":"OR","children":[{"op":"LEAF","expert":"CLIP","query":"wearing blue apron"},{"op":"LEAF","expert":"CLIP","query":"wearing red jacket"},{"op":"LEAF","expert":"CLIP","query":"wearing green hat"}]}]}

Example 2 - Shot ordering question (detect each shot independently, NO SEQ):
Q: "The video includes 4 shots: (1) A dog running on a beach (2) A sunset over the ocean (3) A woman surfing a wave (4) A lifeguard tower on sand. Select the correct order." Options: ["3->1->4->2", "2->4->1->3", "1->2->3->4"]
{"op":"OR","children":[{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"dog"},{"op":"LEAF","expert":"CLIP","query":"dog running on sandy beach"}]},{"op":"AND","children":[{"op":"LEAF","expert":"CLIP","query":"sunset over ocean"},{"op":"LEAF","expert":"CLIP","query":"orange sky reflected on water"}]},{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"surfer"},{"op":"LEAF","expert":"CLIP","query":"woman surfing a wave"}]},{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"lifeguard tower"},{"op":"LEAF","expert":"CLIP","query":"lifeguard tower on sandy beach"}]}]}

Example 3 - OCR + Visual:
Q: "What warning label is shown on the chemical bottle?" Options: ["Flammable", "Corrosive"]
{"op":"AND","children":[{"op":"LEAF","expert":"OVD","query":"chemical bottle"},{"op":"LEAF","expert":"CLIP","query":"warning label on bottle"},{"op":"OR","children":[{"op":"AND","children":[{"op":"LEAF","expert":"OCR","query":"flammable"},{"op":"LEAF","expert":"CLIP","query":"fire hazard symbol"}]},{"op":"AND","children":[{"op":"LEAF","expert":"OCR","query":"corrosive"},{"op":"LEAF","expert":"CLIP","query":"corrosion hazard symbol"}]}]}]}"""

    return examples


def _build_system_prompt(use_asr: bool = True, use_clap: bool = True) -> str:
    """Build the system prompt with conditional audio expert sections."""
    expert_literals = _expert_literals_str(use_asr, use_clap)

    parts = []

    parts.append("""You are a Neuro-Symbolic Logic Parser for a Multimodal Video Understanding system.
Convert Video QA questions into structured logical trees that leverage ALL relevant modalities.

### THE EXPERTS

**Visual Experts:**
1. **OVD** - Open-Vocabulary Object Detection (YOLO-World)
   - For: Physical objects, people, and visual attributes
   - Examples: "person", "car", "dog", "red car", "man in suit", "white dog"
   - Supports attribute+noun phrases. Never include numbers, counts, or proper names.
   - OVD CANNOT detect: countries, teams, names, states, emotions, actions, text, numbers

2. **OCR** - On-Screen Text Recognition
   - For: Text visible on screen - signs, labels, jersey numbers, names, scoreboards
   - Examples: "Exit", "10", "Warning", "Korea"
   - Use for team/country names that appear as text on jerseys, banners, or scoreboards

3. **CLIP** - Semantic Visual Understanding
   - For: Actions, scenes, visual states, atmosphere, abstract visual concepts
   - Examples: "person running", "cooking", "sunset", "player injured", "crowded room"
   - Use for visual states like "injured", "celebrating", "angry", clothing descriptions like "blue shirt"
   - CLIP is VISUAL ONLY - queries must describe something you can SEE in a video frame
   - Good: "person speaking", "diagram on screen", "cooking", "crowded room"
   - Bad: "quantum entanglement", "probability wave", "innovation" (invisible concepts) """)

    if use_asr or use_clap:
        parts.append("**Audio Experts:**")

    if use_asr:
        parts.append("""4. **ASR** - Speech Recognition
   - For: Spoken words, dialogue, narration, verbal references
   - Use SHORT keywords (1-3 words), never full sentences
   - Examples: "reaction", "careful", "goal", "Korea"
   - CRITICAL: People TALK about what is shown. When a question mentions an object/action/topic, add an ASR leaf with related spoken keywords alongside visual leaves.""")

    if use_clap:
        parts.append("""5. **CLAP** - Environmental Audio Events
   - For: Non-speech sounds, music, sound effects, ambient audio
   - Examples: "doorbell ringing", "applause", "glass breaking", "whistle", "crowd cheering" """)

    # Operators
    and_combine = "\n  - Combine visual + audio evidence for the same event" if (use_asr or use_clap) else ""
    parts.append(f"""
### THE OPERATORS

- **AND**: All children must co-occur (same frame). Use to:
  - Decompose complex descriptions into atomic parts{and_combine}

- **OR**: At least one child satisfied. Use for:
  - Combining multiple-choice options
  - Alternative ways to detect the same thing

- **SEQ**: Temporal sequence (earliest first, latest last). ONLY when order is EXPLICITLY STATED:
  - "after X happens" -> SEQ(X, Y)
  - "first X, then Y" -> SEQ(X, Y)
  - "X happens before Y" -> SEQ(X, Y)
  - IMPORTANT: If the question ASKS about order or sequence of shots/scenes, do NOT use SEQ anywhere in the tree. Instead:
    * Build ONE AND node per shot/scene description to detect it independently
    * Wrap all shot AND-nodes in a single flat OR: OR(AND(shot1), AND(shot2), ...)
    * Do NOT duplicate shots across answer permutations — each shot appears exactly once
    * Temporal ordering is determined at inference time from frame timestamps, not from tree structure
    * These phrases MUST trigger this pattern:
      "In which order", "What is the order", "What is the correct order",
      "Select the option that correctly reflects the order",
      "Which of the following is the correct order",
      "What is the sequence", "In what sequence"

- **RIGHT_AFTER**: Immediate temporal succession. Exactly 2 children [cause, effect].
  - "right after X" -> RIGHT_AFTER(X, Y)""")

    # Key rules
    rules = "\n### KEY RULES\n"
    if use_asr or use_clap:
        rules += "\n1. **MULTIMODAL**: Each MCQ option should combine visual AND audio evidence when possible. Never make a tree with only one expert type. Even for narration/documentary questions, always include visual experts (CLIP for scenes, OCR for on-screen text, OVD for visible objects) alongside ASR."
    else:
        rules += "\n1. **MULTIMODAL**: Use multiple visual experts when possible. Never make a tree with only one expert type."

    if use_asr:
        rules += "\n2. **ASR OVERLAP**: Add ASR leaves with short keywords alongside visual leaves - narrators often describe what is shown."

    rules += "\n3. **MCQ STRUCTURE**: AND(shared_context, OR(option_1, option_2, ...)) - factor shared elements OUT of the OR. ALL options MUST be inside the OR."
    rules += '\n4. **DECOMPOSE RICH DESCRIPTIONS**: When a scene/option describes multiple elements (person + attributes + objects + setting), create separate leaves for each: OVD for objects/people (e.g. "bald man", "large weapon"), CLIP for settings/states (e.g. "dark futuristic setting", "ornate structure"). Extract ALL key visual details, not just one summary leaf.'
    rules += "\n5. **SEQ ONLY FOR KNOWN ORDER**: Only use SEQ when the question explicitly states the order (\"after X\", \"first X then Y\"). If the question ASKS about ordering of shots/scenes, do NOT use SEQ anywhere — build one AND per shot and wrap in a flat OR. Each shot appears exactly once."

    if use_asr:
        rules += "\n6. **NAMES -> OCR + ASR**: Any proper names (people, teams, characters, countries) -> OCR (text on screen) + ASR (spoken). For character appearances, also add CLIP for visual description. Never use OVD for names."
    else:
        rules += "\n6. **NAMES -> OCR**: Any proper names (people, teams, characters, countries) -> OCR (text on screen). For character appearances, also add CLIP for visual description. Never use OVD for names."

    rules += '\n7. **VISUAL STATES -> CLIP**: States like "injured", "celebrating", "sleeping" -> CLIP, not ASR alone.'
    rules += '\n8. **META-OPTIONS**: Options like "Same", "All of the above", "Cannot be determined", "Not mentioned" are NOT detectable - ALWAYS skip them in the OR. Only include options that describe observable events/objects.'
    rules += '\n9. **ACTIONS IN OPTIONS**: When options describe actions (e.g. "driving a car", "spraying perfume"), use AND(OVD:object, CLIP:action) per option, not OVD alone.'
    rules += "\n10. **TEMPORAL CAUSE**: In SEQ/RIGHT_AFTER, the cause child should be the triggering ACTION (CLIP), not just a person (OVD)."
    rules += '\n11. **OVERLAPPING EXPERTS**: It is encouraged to use multiple experts on overlapping subjects. E.g., OVD:"basketball player" + CLIP:"basketball player scoring with a dunk" — OVD detects the person while CLIP captures the action in context.'
    if use_asr or use_clap:
        rules += '\n12. **VISUAL GROUNDING**: Never build ASR-only options. For each OR branch, always pair ASR with at least one visual expert. For abstract topics, add CLIP for the visible scene (e.g. "person speaking", "diagram on screen"), OCR for on-screen text, or OVD for visible objects.'
    if use_asr:
        rules += '\n13. **OVERLAPPING PREDICATES**: Mix experts with overlapping terms for robust detection. E.g., for "placing green onions on rack": OVD:"rack", CLIP:"placing green onions on rack", ASR:"rack", ASR:"placed".'
        rules += '\n14. **SUBTITLES -> ASR**: When the question refers to subtitles, use ASR expert to identify the text among spoken words, in addition to OCR for on-screen text.'
    parts.append(rules)

    # Output format
    parts.append(f"""
### OUTPUT FORMAT
Return a single JSON object:
{{"op": "AND"|"OR"|"SEQ"|"RIGHT_AFTER"|"LEAF", "children": [...], "expert": {expert_literals}, "query": "string"}}
- "expert" and "query" only for LEAF nodes. "children" only for non-LEAF nodes.""")

    # Examples
    parts.append("\n### EXAMPLES")
    parts.append(_build_examples(use_asr, use_clap))

    parts.append('\nReturn ONLY a single valid JSON object. No markdown, no explanation, no extra text.')

    return "\n".join(parts)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def parse_query_to_tree(self, query: str, enabled_audio_experts: Optional[Dict[str, bool]] = None, temperature: Optional[float] = None) -> dict:
        """
        Parse a natural language query into a logical tree structure.

        Args:
            query: Natural language query string
            enabled_audio_experts: Optional dict indicating which audio experts are available.
                Example: {"ASR": True, "CLAP": False}. Defaults to both enabled.
            temperature: Optional temperature override for retry with sampling.
                None = use default (deterministic/greedy).

        Returns:
            Dictionary representation of the LogicalTreeNode
        """
        pass

    @staticmethod
    def _resolve_audio_flags(enabled_audio_experts: Optional[Dict[str, bool]] = None):
        """Resolve audio expert flags with defaults."""
        if enabled_audio_experts is None:
            return True, True
        return enabled_audio_experts.get("ASR", True), enabled_audio_experts.get("CLAP", True)


class OpenAILLM(BaseLLM):
    """OpenAI GPT-4 implementation with structured outputs."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", seed: int = 42):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.seed = seed

    def parse_query_to_tree(self, query: str, enabled_audio_experts: Optional[Dict[str, bool]] = None, temperature: Optional[float] = None) -> dict:
        """Parse query using OpenAI with structured outputs."""
        use_asr, use_clap = self._resolve_audio_flags(enabled_audio_experts)
        system_prompt = _build_system_prompt(use_asr, use_clap)
        temp = temperature if temperature is not None else 0.0

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                response_format=LogicalTreeNode,
                temperature=temp,
                seed=self.seed
            )

            tree_node = response.choices[0].message.parsed
            return tree_node.model_dump()

        except Exception as e:
            raise RuntimeError(f"Failed to parse query with OpenAI: {str(e)}")

    def parse_query_to_tree_fallback(self, query: str, enabled_audio_experts: Optional[Dict[str, bool]] = None) -> dict:
        """Fallback method using JSON mode instead of structured outputs."""
        use_asr, use_clap = self._resolve_audio_flags(enabled_audio_experts)
        expert_literals = _expert_literals_str(use_asr, use_clap)

        system_prompt = f"""You are a Neuro-Symbolic Logic Parser for a Computer Vision system.

Output a JSON object following this schema:
- op: "AND" | "OR" | "SEQ" | "LEAF"
- expert: {expert_literals} (only for LEAF nodes)
- query: string (only for LEAF nodes)
- children: array of nodes (only for non-LEAF nodes)

Choose experts wisely:
- OVD: Specific, countable physical objects or people (e.g., "blue shirt", "black car", "man")
- OCR: Specific text on screen, subtitles, or signs (e.g., "Plans", "upcoming plans")
- CLIP: Actions, scene descriptions, atmosphere, or abstract concepts (e.g., "walking", "sunset", "argument")"""

        if use_asr:
            system_prompt += '\n- ASR: Spoken words, dialogue, narration (e.g., "be careful", "next step")'
        if use_clap:
            system_prompt += '\n- CLAP: Non-speech sounds, audio events (e.g., "doorbell ringing", "applause")'

        system_prompt += """

Operators:
- AND: Visual conjunction - both things in SAME frame
- OR: Disjunction - only one of the things in the same frame, also used for multiple-choice options
- SEQ: Temporal sequence - CRITICAL: order children chronologically (earliest first, latest last)

Return only valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                seed=self.seed
            )

            tree_dict = json.loads(response.choices[0].message.content)
            tree_dict = repair_tree(tree_dict)
            tree_node = LogicalTreeNode(**tree_dict)
            return tree_node.model_dump()

        except Exception as e:
            raise RuntimeError(f"Failed to parse query with OpenAI (fallback): {str(e)}")


class Qwen3VLLLM(BaseLLM):
    """Local Qwen3-VL implementation using HuggingFace transformers (text-only mode)."""

    def __init__(self, model: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda:0"):
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError:
            raise ImportError("Please install transformers and torch")

        self.model_name = model
        self.device = device

        self.processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        ).eval()

    def parse_query_to_tree(self, query: str, enabled_audio_experts: Optional[Dict[str, bool]] = None, temperature: Optional[float] = None) -> dict:
        """Parse query using local Qwen3-VL model in text-only mode."""
        import gc
        import re
        import torch

        use_asr, use_clap = self._resolve_audio_flags(enabled_audio_experts)
        system_prompt = _build_system_prompt(use_asr, use_clap)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": f"Query: {query}"}]},
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Use sampling when temperature is provided, otherwise greedy
        if temperature is not None:
            gen_kwargs = dict(max_new_tokens=2048, temperature=temperature, top_p=0.9, do_sample=True)
        else:
            gen_kwargs = dict(max_new_tokens=2048, do_sample=False)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            input_len = inputs["input_ids"].shape[1]
            generated = [out[input_len:] for out in output_ids]
            response_text = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # Strip any leftover <think>...</think> tags
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

            # Extract JSON from response (handle potential markdown wrapping)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Extract first complete JSON object (handle trailing text/explanations)
            brace_count = 0
            json_end = -1
            for idx, ch in enumerate(response_text):
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = idx + 1
                        break
            if json_end > 0:
                response_text = response_text[:json_end]

            tree_dict = json.loads(response_text)
            tree_dict = repair_tree(tree_dict)
            tree_node = LogicalTreeNode(**tree_dict)
            return tree_node.model_dump()

        except Exception as e:
            raise RuntimeError(f"Failed to parse query with Qwen3-VL: {str(e)}")
        finally:
            del inputs
            torch.cuda.empty_cache()
            gc.collect()


def create_llm(provider: str = "openai", model: Optional[str] = None, api_key: Optional[str] = None, device: Optional[str] = None, seed: int = 42) -> BaseLLM:
    """
    Factory function to create an LLM instance.

    Args:
        provider: LLM provider ("openai" or "qwen3vl")
        model: Model name (defaults to provider's default)
        api_key: API key (defaults to env var, not used for qwen3vl)
        device: Device for local models (qwen3vl only, defaults to "cuda:0")
        seed: Seed for deterministic outputs (OpenAI only)

    Returns:
        BaseLLM instance
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAILLM(api_key=api_key, model=model or "gpt-4o", seed=seed)
    elif provider == "qwen3vl":
        return Qwen3VLLLM(model=model or "Qwen/Qwen3-VL-8B-Instruct", device=device or "cuda:0")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Supported: 'openai', 'qwen3vl'")
