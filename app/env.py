"""
ClinicalDecisionEnv: Main environment class implementing OpenEnv spec.
Provides step(), reset(), state() API for clinical decision-making simulation.
"""

import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.models import (
    Observation, Action, Reward, StepResult, EnvironmentState,
    ActionType, PatientHistory, PatientVitals, LabResult
)
from app.tasks import TaskRegistry, TaskGrader
from app.rewards import RewardCalculator


class ClinicalDecisionEnv:
    """
    AI Clinical Decision Support Environment.

    Simulates a clinical decision-making scenario where an AI agent
    must diagnose patients and recommend treatments through multi-step
    interactions including history taking, test ordering, and treatment planning.
    """

    def __init__(
        self,
        task_id: str = "easy_diagnosis",
        case_id: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the clinical environment.

        Args:
            task_id: Task difficulty level (easy_diagnosis, medium_diagnosis, hard_diagnosis)
            case_id: Specific case ID to load (optional, random if not specified)
            seed: Random seed for reproducibility
        """
        self.task_registry = TaskRegistry()
        self.task_id = task_id
        self.specific_case_id = case_id
        self.seed = seed

        # Episode state
        self.episode_id: Optional[str] = None
        self.current_case: Optional[Dict[str, Any]] = None
        self.observation: Optional[Observation] = None
        self.reward_calculator: Optional[RewardCalculator] = None
        self.grader: Optional[TaskGrader] = None
        self.episode_done: bool = False
        self.step_count: int = 0

        # Action history for grading
        self.actions_taken: List[Dict[str, Any]] = []
        self.tests_ordered: List[str] = []
        self.questions_asked: List[str] = []
        self.diagnoses_suggested: List[str] = []
        self.treatments_recommended: List[str] = []

    def reset(self, task_id: Optional[str] = None, case_id: Optional[str] = None) -> Observation:
        """
        Reset environment to initial state.

        Args:
            task_id: Optionally change task
            case_id: Optionally specify case

        Returns:
            Initial observation
        """
        if task_id:
            self.task_id = task_id
        if case_id:
            self.specific_case_id = case_id

        # Get task and case
        task = self.task_registry.get_task(self.task_id)
        if not task:
            raise ValueError(f"Unknown task: {self.task_id}")

        if self.specific_case_id:
            self.current_case = self.task_registry.get_case_by_id(self.specific_case_id)
        else:
            self.current_case = self.task_registry.get_random_case(self.task_id)

        if not self.current_case:
            raise ValueError(f"No case found for task: {self.task_id}")

        # Initialize episode
        self.episode_id = str(uuid.uuid4())[:8]
        self.episode_done = False
        self.step_count = 0

        # Clear history
        self.actions_taken = []
        self.tests_ordered = []
        self.questions_asked = []
        self.diagnoses_suggested = []
        self.treatments_recommended = []

        # Initialize reward calculator and grader
        ground_truth = self.current_case.get("ground_truth", {})
        self.reward_calculator = RewardCalculator(
            ground_truth=ground_truth,
            case_difficulty=task.difficulty
        )
        self.grader = self.task_registry.create_grader(self.task_id, self.current_case)

        # Build initial observation
        self.observation = self._build_initial_observation(task.max_steps)

        return self.observation

    def step(self, action: Action) -> StepResult:
        """
        Execute an action in the environment.

        Args:
            action: Action to execute (ask_question, order_test, suggest_diagnosis, etc.)

        Returns:
            StepResult containing observation, reward, done, info
        """
        if self.episode_done:
            return StepResult(
                observation=self.observation,
                reward=Reward(total=0.0, explanation="Episode already complete"),
                done=True,
                info={"error": "Episode already complete, call reset()"}
            )

        if not self.current_case or not self.observation:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.step_count += 1

        # Record action
        self.actions_taken.append({
            "step": self.step_count,
            "action_type": action.action_type,
            "action_value": action.action_value,
            "reasoning": action.reasoning
        })

        # Process action and get feedback
        feedback, error = self._process_action(action)

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            action_type=ActionType(action.action_type),
            action_value=action.action_value,
            step_number=self.step_count,
            episode_done=action.action_type == ActionType.FINALIZE_CASE
        )

        # Check for episode end conditions
        task = self.task_registry.get_task(self.task_id)
        if (action.action_type == ActionType.FINALIZE_CASE or
            self.step_count >= task.max_steps):
            self.episode_done = True

        # Update observation
        self.observation.step_number = self.step_count
        self.observation.last_action_feedback = feedback
        self.observation.last_action_error = error
        self.observation.current_diagnoses = self.diagnoses_suggested.copy()
        self.observation.current_treatments = self.treatments_recommended.copy()

        # Build info
        info = {
            "step": self.step_count,
            "max_steps": task.max_steps,
            "action_processed": action.action_type
        }

        if self.episode_done:
            # Run grader and include final score
            grade_result = self.grader.grade_episode(
                actions_taken=self.actions_taken,
                final_diagnoses=self.diagnoses_suggested,
                final_treatments=self.treatments_recommended,
                tests_ordered=self.tests_ordered,
                questions_asked=self.questions_asked,
                steps_used=self.step_count
            )
            info["final_grade"] = grade_result

        return StepResult(
            observation=self.observation,
            reward=reward,
            done=self.episode_done,
            info=info
        )

    def state(self) -> EnvironmentState:
        """
        Get current environment state.

        Returns:
            Full environment state including ground truth (for debugging/analysis)
        """
        ground_truth = self.current_case.get("ground_truth", {}) if self.current_case else {}
        task = self.task_registry.get_task(self.task_id)

        final_score = None
        if self.episode_done and self.grader:
            grade_result = self.grader.grade_episode(
                actions_taken=self.actions_taken,
                final_diagnoses=self.diagnoses_suggested,
                final_treatments=self.treatments_recommended,
                tests_ordered=self.tests_ordered,
                questions_asked=self.questions_asked,
                steps_used=self.step_count
            )
            final_score = grade_result["score"]

        return EnvironmentState(
            observation=self.observation,
            ground_truth_diagnosis=ground_truth.get("primary_diagnosis", []),
            ground_truth_treatment=ground_truth.get("appropriate_treatments", []),
            required_tests=ground_truth.get("required_tests", []),
            critical_questions=ground_truth.get("critical_questions", []),
            case_difficulty=task.difficulty if task else "unknown",
            episode_complete=self.episode_done,
            final_score=final_score
        )

    def close(self):
        """Clean up environment resources."""
        self.current_case = None
        self.observation = None
        self.episode_done = True

    def _build_initial_observation(self, max_steps: int) -> Observation:
        """Build initial observation from case data."""
        case = self.current_case

        # Build patient history
        hist_data = case.get("patient_history", {})
        patient_history = PatientHistory(
            age=hist_data.get("age", 0),
            sex=hist_data.get("sex", "Unknown"),
            chronic_conditions=hist_data.get("chronic_conditions", []),
            allergies=hist_data.get("allergies", []),
            current_medications=hist_data.get("current_medications", []),
            family_history=hist_data.get("family_history", []),
            social_history=hist_data.get("social_history")
        )

        # Build vitals if present
        vitals = None
        if "vitals" in case:
            v = case["vitals"]
            vitals = PatientVitals(
                temperature=v.get("temperature"),
                blood_pressure_systolic=v.get("blood_pressure_systolic"),
                blood_pressure_diastolic=v.get("blood_pressure_diastolic"),
                heart_rate=v.get("heart_rate"),
                respiratory_rate=v.get("respiratory_rate"),
                oxygen_saturation=v.get("oxygen_saturation")
            )

        return Observation(
            episode_id=self.episode_id,
            step_number=0,
            max_steps=max_steps,
            chief_complaint=case.get("chief_complaint", ""),
            presenting_symptoms=case.get("presenting_symptoms", []),
            symptom_duration=case.get("symptom_duration"),
            patient_history=patient_history,
            vitals=vitals,
            lab_results=[],
            imaging_results=[],
            question_answers={},
            consultation_notes=[],
            current_diagnoses=[],
            current_treatments=[],
            last_action_feedback=None,
            last_action_error=None,
            urgency_level=case.get("urgency_level", "routine"),
            risk_flags=case.get("ground_truth", {}).get("red_flags", [])
        )

    def _process_action(self, action: Action) -> tuple:
        """
        Process an action and return feedback/error.

        Returns:
            Tuple of (feedback_string, error_string or None)
        """
        action_type = ActionType(action.action_type)
        action_value = action.action_value.strip()
        available_answers = self.current_case.get("available_question_answers", {})
        available_tests = self.current_case.get("available_test_results", {})

        if action_type == ActionType.ASK_QUESTION:
            self.questions_asked.append(action_value)

            # Find matching answer
            answer = self._find_matching_answer(action_value, available_answers)
            if answer:
                self.observation.question_answers[action_value] = answer
                return f"Patient response: {answer}", None
            else:
                return "Patient: I'm not sure how to answer that.", None

        elif action_type == ActionType.ORDER_TEST:
            self.tests_ordered.append(action_value)

            # Find matching test result
            test_result = self._find_matching_test(action_value, available_tests)
            if test_result:
                lab_result = LabResult(
                    test_name=test_result["test_name"],
                    value=test_result["value"],
                    unit=test_result.get("unit"),
                    reference_range=test_result.get("reference_range"),
                    is_abnormal=test_result.get("is_abnormal", False)
                )
                self.observation.lab_results.append(lab_result)
                abnormal_flag = " [ABNORMAL]" if lab_result.is_abnormal else ""
                return f"Test result - {lab_result.test_name}: {lab_result.value}{abnormal_flag}", None
            else:
                return f"Test '{action_value}' is not available or not applicable.", None

        elif action_type == ActionType.SUGGEST_DIAGNOSIS:
            self.diagnoses_suggested.append(action_value)
            return f"Diagnosis recorded: {action_value}", None

        elif action_type == ActionType.RECOMMEND_TREATMENT:
            self.treatments_recommended.append(action_value)
            return f"Treatment recommendation recorded: {action_value}", None

        elif action_type == ActionType.REQUEST_CONSULTATION:
            self.observation.consultation_notes.append(
                f"Consultation requested: {action_value}"
            )
            return f"Consultation with {action_value} requested.", None

        elif action_type == ActionType.FINALIZE_CASE:
            return "Case finalized. Episode complete.", None

        return "Action processed.", None

    def _find_matching_answer(
        self,
        question: str,
        available_answers: Dict[str, str]
    ) -> Optional[str]:
        """Find a matching answer for a question using fuzzy matching."""
        question_lower = question.lower()

        # Direct match
        for key, answer in available_answers.items():
            if key.lower() in question_lower or question_lower in key.lower():
                return answer

        # Keyword matching
        for key, answer in available_answers.items():
            key_words = set(key.lower().split())
            question_words = set(question_lower.split())
            if len(key_words.intersection(question_words)) >= 2:
                return answer

        return None

    def _find_matching_test(
        self,
        test_name: str,
        available_tests: Dict[str, Dict]
    ) -> Optional[Dict]:
        """Find a matching test result using fuzzy matching."""
        test_lower = test_name.lower()

        # Direct match
        for key, result in available_tests.items():
            if key.lower() in test_lower or test_lower in key.lower():
                return result

        # Abbreviation matching
        abbreviations = {
            "ecg": "ECG", "ekg": "ECG",
            "cbc": "complete blood count",
            "bmp": "basic metabolic panel",
            "cmp": "comprehensive metabolic panel",
            "ua": "urinalysis",
            "lp": "lumbar puncture",
            "cxr": "chest x-ray"
        }

        expanded = abbreviations.get(test_lower, test_lower)
        for key, result in available_tests.items():
            if expanded.lower() in key.lower() or key.lower() in expanded.lower():
                return result

        return None

    def get_action_space(self) -> Dict[str, Any]:
        """Return description of valid actions."""
        return {
            "action_types": [
                {
                    "type": ActionType.ASK_QUESTION.value,
                    "description": "Ask the patient a question",
                    "example_value": "How long have you had these symptoms?"
                },
                {
                    "type": ActionType.ORDER_TEST.value,
                    "description": "Order a diagnostic test",
                    "example_value": "complete blood count"
                },
                {
                    "type": ActionType.SUGGEST_DIAGNOSIS.value,
                    "description": "Suggest a diagnosis",
                    "example_value": "streptococcal pharyngitis"
                },
                {
                    "type": ActionType.RECOMMEND_TREATMENT.value,
                    "description": "Recommend a treatment",
                    "example_value": "amoxicillin 500mg twice daily for 10 days"
                },
                {
                    "type": ActionType.REQUEST_CONSULTATION.value,
                    "description": "Request specialist consultation",
                    "example_value": "cardiology"
                },
                {
                    "type": ActionType.FINALIZE_CASE.value,
                    "description": "Finalize case and end episode",
                    "example_value": "case complete"
                }
            ]
        }

    def get_observation_space(self) -> Dict[str, Any]:
        """Return description of observation format."""
        return {
            "fields": [
                {"name": "episode_id", "type": "string", "description": "Unique episode identifier"},
                {"name": "step_number", "type": "int", "description": "Current step number"},
                {"name": "max_steps", "type": "int", "description": "Maximum allowed steps"},
                {"name": "chief_complaint", "type": "string", "description": "Patient's main complaint"},
                {"name": "presenting_symptoms", "type": "list[string]", "description": "Initial symptoms"},
                {"name": "patient_history", "type": "PatientHistory", "description": "Patient medical history"},
                {"name": "vitals", "type": "PatientVitals", "description": "Current vital signs"},
                {"name": "lab_results", "type": "list[LabResult]", "description": "Ordered test results"},
                {"name": "question_answers", "type": "dict", "description": "Questions asked and answers"},
                {"name": "current_diagnoses", "type": "list[string]", "description": "Suggested diagnoses"},
                {"name": "current_treatments", "type": "list[string]", "description": "Recommended treatments"},
                {"name": "urgency_level", "type": "string", "description": "Case urgency level"},
                {"name": "last_action_feedback", "type": "string", "description": "Feedback from last action"}
            ]
        }
