"""
Task definitions and graders for the Clinical Decision Support Environment.
Provides 3 difficulty levels: easy, medium, hard with deterministic grading.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.models import ActionType
from app.rewards import RewardCalculator


class TaskDefinition(BaseModel):
    """Definition of a task for the environment."""
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    description: str = Field(..., description="Task description")
    difficulty: str = Field(..., description="easy/medium/hard")
    case_ids: List[str] = Field(..., description="Case IDs for this task")
    max_steps: int = Field(20, description="Maximum steps allowed")
    passing_score: float = Field(0.6, description="Minimum score to pass")
    objective: str = Field(..., description="Clear objective statement")
    success_criteria: List[str] = Field(..., description="Success criteria list")


class TaskGrader:
    """
    Deterministic grader for clinical decision tasks.
    Produces scores between 0.0 and 1.0.
    """

    def __init__(self, task: TaskDefinition, ground_truth: Dict[str, Any]):
        self.task = task
        self.ground_truth = ground_truth
        self.reward_calculator = RewardCalculator(
            ground_truth=ground_truth,
            case_difficulty=task.difficulty
        )

    def grade_episode(
        self,
        actions_taken: List[Dict[str, Any]],
        final_diagnoses: List[str],
        final_treatments: List[str],
        tests_ordered: List[str],
        questions_asked: List[str],
        steps_used: int
    ) -> Dict[str, Any]:
        """
        Grade a complete episode.

        Returns:
            Dict with 'score' (0.0-1.0), 'passed', 'breakdown', 'feedback'
        """
        # Update reward calculator state from episode history
        self.reward_calculator.diagnoses_suggested = set(d.lower() for d in final_diagnoses)
        self.reward_calculator.treatments_recommended = set(t.lower() for t in final_treatments)
        self.reward_calculator.tests_ordered = set(t.lower() for t in tests_ordered)
        self.reward_calculator.questions_asked = set(q.lower() for q in questions_asked)
        self.reward_calculator.total_steps = steps_used

        # Calculate final score
        score = self.reward_calculator.get_final_score()

        # Build detailed breakdown
        breakdown = self._build_breakdown(
            final_diagnoses, final_treatments, tests_ordered, questions_asked, steps_used
        )

        # Generate feedback
        feedback = self._generate_feedback(breakdown)

        return {
            "score": round(score, 3),
            "passed": score >= self.task.passing_score,
            "breakdown": breakdown,
            "feedback": feedback,
            "steps_used": steps_used,
            "max_steps": self.task.max_steps
        }

    def _build_breakdown(
        self,
        diagnoses: List[str],
        treatments: List[str],
        tests: List[str],
        questions: List[str],
        steps: int
    ) -> Dict[str, Any]:
        """Build detailed scoring breakdown."""
        correct_diagnoses = set(d.lower() for d in self.ground_truth.get("primary_diagnosis", []))
        required_tests = set(t.lower() for t in self.ground_truth.get("required_tests", []))
        critical_questions = set(q.lower() for q in self.ground_truth.get("critical_questions", []))
        appropriate_treatments = set(t.lower() for t in self.ground_truth.get("appropriate_treatments", []))
        contraindicated = set(t.lower() for t in self.ground_truth.get("contraindicated_treatments", []))

        # Diagnosis accuracy
        diagnoses_lower = set(d.lower() for d in diagnoses)
        correct_dx = any(
            self.reward_calculator._fuzzy_match(d, c)
            for d in diagnoses_lower
            for c in correct_diagnoses
        )

        # Tests coverage
        tests_lower = set(t.lower() for t in tests)
        tests_hit = sum(
            1 for req in required_tests
            if any(self.reward_calculator._fuzzy_match(t, req) for t in tests_lower)
        )
        tests_coverage = tests_hit / max(len(required_tests), 1)

        # Questions coverage
        questions_lower = set(q.lower() for q in questions)
        questions_hit = sum(
            1 for crit in critical_questions
            if any(self.reward_calculator._fuzzy_match(q, crit) for q in questions_lower)
        )
        questions_coverage = questions_hit / max(len(critical_questions), 1)

        # Treatment appropriateness
        treatments_lower = set(t.lower() for t in treatments)
        treatment_ok = any(
            self.reward_calculator._fuzzy_match(t, a)
            for t in treatments_lower
            for a in appropriate_treatments
        )

        # Safety check
        unsafe = any(
            self.reward_calculator._fuzzy_match(t, c)
            for t in treatments_lower
            for c in contraindicated
        )

        return {
            "diagnosis": {
                "correct": correct_dx,
                "suggested": list(diagnoses),
                "expected": list(self.ground_truth.get("primary_diagnosis", [])),
                "score": 0.4 if correct_dx else 0.0
            },
            "tests": {
                "ordered": list(tests),
                "required": list(required_tests),
                "coverage": round(tests_coverage, 2),
                "score": round(0.2 * tests_coverage, 3)
            },
            "questions": {
                "asked": list(questions),
                "critical": list(critical_questions),
                "coverage": round(questions_coverage, 2),
                "score": round(0.15 * questions_coverage, 3)
            },
            "treatment": {
                "recommended": list(treatments),
                "appropriate": list(appropriate_treatments),
                "is_appropriate": treatment_ok,
                "score": 0.15 if treatment_ok else 0.0
            },
            "safety": {
                "unsafe_treatments": unsafe,
                "contraindicated": list(contraindicated),
                "score": 0.1 if not unsafe else 0.0
            },
            "efficiency": {
                "steps_used": steps,
                "max_steps": self.task.max_steps
            }
        }

    def _generate_feedback(self, breakdown: Dict[str, Any]) -> List[str]:
        """Generate human-readable feedback."""
        feedback = []

        if breakdown["diagnosis"]["correct"]:
            feedback.append("Correct diagnosis reached.")
        else:
            expected = breakdown["diagnosis"]["expected"]
            feedback.append(f"Incorrect diagnosis. Expected: {', '.join(expected)}")

        tests_cov = breakdown["tests"]["coverage"]
        if tests_cov >= 0.8:
            feedback.append("Excellent test coverage.")
        elif tests_cov >= 0.5:
            feedback.append("Moderate test coverage - some key tests missing.")
        else:
            feedback.append(f"Insufficient tests. Required: {', '.join(breakdown['tests']['required'])}")

        q_cov = breakdown["questions"]["coverage"]
        if q_cov >= 0.8:
            feedback.append("Good history taking.")
        elif q_cov < 0.5:
            feedback.append("Important questions not asked.")

        if breakdown["treatment"]["is_appropriate"]:
            feedback.append("Appropriate treatment plan.")
        else:
            feedback.append("Treatment plan needs improvement.")

        if breakdown["safety"]["unsafe_treatments"]:
            feedback.append("SAFETY CONCERN: Contraindicated treatment recommended!")

        return feedback


class TaskRegistry:
    """Registry of all available tasks."""

    def __init__(self, cases_path: Optional[Path] = None):
        if cases_path is None:
            cases_path = Path(__file__).parent / "cases" / "cases.json"

        with open(cases_path) as f:
            self.cases_data = json.load(f)

        self.tasks = self._build_tasks()

    def _build_tasks(self) -> Dict[str, TaskDefinition]:
        """Build task definitions from case data."""
        tasks = {}

        # Easy Task: Basic Diagnosis
        tasks["easy_diagnosis"] = TaskDefinition(
            task_id="easy_diagnosis",
            name="Basic Diagnosis Challenge",
            description="Diagnose common conditions with clear presentations. "
                        "Straightforward cases with obvious symptoms and standard workup.",
            difficulty="easy",
            case_ids=[c["case_id"] for c in self.cases_data["easy_cases"]],
            max_steps=15,
            passing_score=0.6,
            objective="Correctly diagnose a patient with a common condition and recommend appropriate treatment.",
            success_criteria=[
                "Identify the correct diagnosis",
                "Order at least one relevant diagnostic test",
                "Ask relevant history questions",
                "Recommend appropriate treatment",
                "Avoid contraindicated treatments"
            ]
        )

        # Medium Task: Complex Diagnosis
        tasks["medium_diagnosis"] = TaskDefinition(
            task_id="medium_diagnosis",
            name="Complex Diagnosis Challenge",
            description="Manage cases requiring multi-step reasoning. "
                        "Emergency presentations, multiple risk factors, and time-sensitive decisions.",
            difficulty="medium",
            case_ids=[c["case_id"] for c in self.cases_data["medium_cases"]],
            max_steps=20,
            passing_score=0.55,
            objective="Navigate a complex medical case with appropriate urgency, "
                      "thorough workup, and correct diagnosis/treatment.",
            success_criteria=[
                "Recognize emergency presentations",
                "Order comprehensive diagnostic workup",
                "Ask critical history questions",
                "Identify correct diagnosis",
                "Initiate appropriate urgent treatment",
                "Consider differential diagnoses"
            ]
        )

        # Hard Task: Challenging Cases
        tasks["hard_diagnosis"] = TaskDefinition(
            task_id="hard_diagnosis",
            name="Expert Diagnosis Challenge",
            description="Handle challenging cases with atypical presentations, "
                        "multiple comorbidities, rare conditions, and diagnostic pitfalls.",
            difficulty="hard",
            case_ids=[c["case_id"] for c in self.cases_data["hard_cases"]],
            max_steps=25,
            passing_score=0.5,
            objective="Successfully manage a complex case that would challenge "
                      "experienced clinicians, considering all complicating factors.",
            success_criteria=[
                "Navigate atypical or misleading presentations",
                "Consider rare but important diagnoses",
                "Account for comorbidities and contraindications",
                "Order specialized/advanced testing",
                "Provide comprehensive treatment plan",
                "Recognize critical situations requiring urgent intervention"
            ]
        )

        return tasks

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all available tasks."""
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "description": t.description,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "passing_score": t.passing_score
            }
            for t in self.tasks.values()
        ]

    def get_random_case(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a random case for a task."""
        task = self.get_task(task_id)
        if not task:
            return None

        difficulty_key = f"{task.difficulty}_cases"
        cases = self.cases_data.get(difficulty_key, [])

        if not cases:
            return None

        return random.choice(cases)

    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific case by its ID."""
        for difficulty in ["easy_cases", "medium_cases", "hard_cases"]:
            for case in self.cases_data.get(difficulty, []):
                if case["case_id"] == case_id:
                    return case
        return None

    def create_grader(self, task_id: str, case: Dict[str, Any]) -> TaskGrader:
        """Create a grader for a specific task and case."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")

        ground_truth = case.get("ground_truth", {})
        return TaskGrader(task=task, ground_truth=ground_truth)
