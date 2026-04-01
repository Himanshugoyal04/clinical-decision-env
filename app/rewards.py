"""
Reward calculation logic for the Clinical Decision Support Environment.
Provides meaningful partial progress signals throughout episodes.
"""

from typing import List, Dict, Any, Set
from app.models import Reward, ActionType


class RewardCalculator:
    """
    Calculates rewards for clinical decision-making actions.
    Rewards partial progress and penalizes unsafe/inefficient behavior.
    """

    def __init__(self, ground_truth: Dict[str, Any], case_difficulty: str = "easy"):
        self.ground_truth = ground_truth
        self.case_difficulty = case_difficulty

        # Extract ground truth components
        self.correct_diagnoses = set(
            d.lower() for d in ground_truth.get("primary_diagnosis", [])
        )
        self.differential_diagnoses = set(
            d.lower() for d in ground_truth.get("differential_diagnoses", [])
        )
        self.required_tests = set(
            t.lower() for t in ground_truth.get("required_tests", [])
        )
        self.critical_questions = set(
            q.lower() for q in ground_truth.get("critical_questions", [])
        )
        self.appropriate_treatments = set(
            t.lower() for t in ground_truth.get("appropriate_treatments", [])
        )
        self.contraindicated_treatments = set(
            t.lower() for t in ground_truth.get("contraindicated_treatments", [])
        )
        self.red_flags = ground_truth.get("red_flags", [])

        # Track cumulative actions for efficiency calculations
        self.tests_ordered: Set[str] = set()
        self.questions_asked: Set[str] = set()
        self.diagnoses_suggested: Set[str] = set()
        self.treatments_recommended: Set[str] = set()

        # Step tracking
        self.total_steps = 0
        self.max_efficient_steps = self._calculate_max_efficient_steps()

    def _calculate_max_efficient_steps(self) -> int:
        """Calculate the expected minimum steps for optimal path."""
        min_steps = len(self.required_tests) + len(self.critical_questions) + 2
        if self.case_difficulty == "medium":
            min_steps += 2
        elif self.case_difficulty == "hard":
            min_steps += 4
        return min_steps

    def calculate_reward(
        self,
        action_type: ActionType,
        action_value: str,
        step_number: int,
        episode_done: bool = False
    ) -> Reward:
        """
        Calculate reward for an action.

        Returns detailed reward breakdown with partial credit.
        """
        self.total_steps = step_number
        action_value_lower = action_value.lower()

        diagnostic_accuracy = 0.0
        information_gathering = 0.0
        treatment_appropriateness = 0.0
        efficiency = 0.0
        safety = 0.0
        explanation_parts = []

        if action_type == ActionType.ASK_QUESTION:
            reward_result = self._evaluate_question(action_value_lower)
            information_gathering = reward_result["reward"]
            explanation_parts.append(reward_result["explanation"])
            self.questions_asked.add(action_value_lower)

        elif action_type == ActionType.ORDER_TEST:
            reward_result = self._evaluate_test(action_value_lower)
            information_gathering = reward_result["reward"]
            explanation_parts.append(reward_result["explanation"])
            self.tests_ordered.add(action_value_lower)

        elif action_type == ActionType.SUGGEST_DIAGNOSIS:
            reward_result = self._evaluate_diagnosis(action_value_lower)
            diagnostic_accuracy = reward_result["reward"]
            explanation_parts.append(reward_result["explanation"])
            self.diagnoses_suggested.add(action_value_lower)

        elif action_type == ActionType.RECOMMEND_TREATMENT:
            reward_result = self._evaluate_treatment(action_value_lower)
            treatment_appropriateness = reward_result["reward"]
            safety = reward_result.get("safety_penalty", 0.0)
            explanation_parts.append(reward_result["explanation"])
            self.treatments_recommended.add(action_value_lower)

        elif action_type == ActionType.REQUEST_CONSULTATION:
            reward_result = self._evaluate_consultation(action_value_lower)
            information_gathering = reward_result["reward"]
            explanation_parts.append(reward_result["explanation"])

        elif action_type == ActionType.FINALIZE_CASE:
            reward_result = self._evaluate_finalization()
            diagnostic_accuracy = reward_result["diagnostic_accuracy"]
            treatment_appropriateness = reward_result["treatment_appropriateness"]
            efficiency = reward_result["efficiency"]
            explanation_parts.append(reward_result["explanation"])

        # Calculate efficiency penalty for excessive steps
        if step_number > self.max_efficient_steps * 1.5:
            inefficiency_penalty = -0.05 * (step_number - self.max_efficient_steps * 1.5)
            efficiency += max(inefficiency_penalty, -0.3)
            explanation_parts.append(f"Inefficiency penalty: {efficiency:.2f}")

        # Bonus for making progress toward required tests/questions
        progress_bonus = self._calculate_progress_bonus()
        if progress_bonus > 0:
            information_gathering += progress_bonus

        total = (
            diagnostic_accuracy +
            information_gathering +
            treatment_appropriateness +
            efficiency +
            safety
        )

        # Clamp total reward
        total = max(-2.0, min(2.0, total))

        return Reward(
            total=round(total, 3),
            diagnostic_accuracy=round(diagnostic_accuracy, 3),
            information_gathering=round(information_gathering, 3),
            treatment_appropriateness=round(treatment_appropriateness, 3),
            efficiency=round(efficiency, 3),
            safety=round(safety, 3),
            explanation=" | ".join(explanation_parts)
        )

    def _evaluate_question(self, question: str) -> Dict[str, Any]:
        """Evaluate a question asked by the agent."""
        # Check if question matches critical questions
        for critical_q in self.critical_questions:
            if self._fuzzy_match(question, critical_q):
                if question not in self.questions_asked:
                    return {
                        "reward": 0.15,
                        "explanation": f"Critical question asked: +0.15"
                    }
                else:
                    return {
                        "reward": -0.05,
                        "explanation": "Duplicate question: -0.05"
                    }

        # Relevant but not critical question
        relevant_keywords = ["duration", "history", "symptom", "pain", "medication", "allergy"]
        if any(kw in question for kw in relevant_keywords):
            if question not in self.questions_asked:
                return {
                    "reward": 0.05,
                    "explanation": "Relevant question asked: +0.05"
                }

        return {
            "reward": 0.0,
            "explanation": "Question asked with neutral impact"
        }

    def _evaluate_test(self, test: str) -> Dict[str, Any]:
        """Evaluate a test ordered by the agent."""
        # Check if test is required
        for required_test in self.required_tests:
            if self._fuzzy_match(test, required_test):
                if test not in self.tests_ordered:
                    return {
                        "reward": 0.2,
                        "explanation": f"Required test ordered: +0.2"
                    }
                else:
                    return {
                        "reward": -0.1,
                        "explanation": "Duplicate test ordered: -0.1"
                    }

        # Check if test is reasonable but not required
        reasonable_tests = [
            "blood", "cbc", "chemistry", "metabolic", "culture", "x-ray",
            "ecg", "ekg", "urinalysis", "ct", "mri"
        ]
        if any(rt in test for rt in reasonable_tests):
            if test not in self.tests_ordered:
                return {
                    "reward": 0.02,
                    "explanation": "Reasonable test ordered: +0.02"
                }

        # Unnecessary test penalty
        if test not in self.tests_ordered:
            return {
                "reward": -0.05,
                "explanation": "Unnecessary test ordered: -0.05"
            }

        return {
            "reward": -0.1,
            "explanation": "Duplicate test: -0.1"
        }

    def _evaluate_diagnosis(self, diagnosis: str) -> Dict[str, Any]:
        """Evaluate a diagnosis suggested by the agent."""
        # Check for correct primary diagnosis
        for correct_dx in self.correct_diagnoses:
            if self._fuzzy_match(diagnosis, correct_dx):
                return {
                    "reward": 0.5,
                    "explanation": f"Correct diagnosis suggested: +0.5"
                }

        # Check for valid differential diagnosis
        for diff_dx in self.differential_diagnoses:
            if self._fuzzy_match(diagnosis, diff_dx):
                return {
                    "reward": 0.15,
                    "explanation": f"Valid differential diagnosis: +0.15"
                }

        # Incorrect diagnosis
        return {
            "reward": -0.2,
            "explanation": "Incorrect diagnosis suggested: -0.2"
        }

    def _evaluate_treatment(self, treatment: str) -> Dict[str, Any]:
        """Evaluate a treatment recommendation."""
        # Check for contraindicated treatment (safety critical)
        for contra in self.contraindicated_treatments:
            if self._fuzzy_match(treatment, contra):
                return {
                    "reward": -0.5,
                    "safety_penalty": -0.5,
                    "explanation": f"DANGEROUS: Contraindicated treatment: -1.0"
                }

        # Check for appropriate treatment
        for appropriate in self.appropriate_treatments:
            if self._fuzzy_match(treatment, appropriate):
                return {
                    "reward": 0.3,
                    "explanation": f"Appropriate treatment recommended: +0.3"
                }

        # Neutral/suboptimal but not dangerous treatment
        return {
            "reward": -0.05,
            "explanation": "Suboptimal treatment recommendation: -0.05"
        }

    def _evaluate_consultation(self, specialty: str) -> Dict[str, Any]:
        """Evaluate a consultation request."""
        # Define appropriate consultations based on case complexity
        appropriate_consults = {
            "easy": ["primary care", "family medicine"],
            "medium": ["cardiology", "neurology", "infectious disease", "internal medicine"],
            "hard": ["gastroenterology", "hepatology", "cardiology", "electrophysiology",
                     "infectious disease", "hematology", "oncology", "surgery", "icu"]
        }

        relevant_consults = appropriate_consults.get(self.case_difficulty, [])
        for consult in relevant_consults:
            if self._fuzzy_match(specialty, consult):
                return {
                    "reward": 0.1,
                    "explanation": f"Appropriate consultation requested: +0.1"
                }

        return {
            "reward": 0.0,
            "explanation": "Consultation requested"
        }

    def _evaluate_finalization(self) -> Dict[str, Any]:
        """Evaluate the final case summary."""
        diagnostic_accuracy = 0.0
        treatment_appropriateness = 0.0
        efficiency = 0.0
        explanation_parts = []

        # Check if correct diagnosis was made
        correct_dx_found = any(
            any(self._fuzzy_match(suggested, correct)
                for correct in self.correct_diagnoses)
            for suggested in self.diagnoses_suggested
        )

        if correct_dx_found:
            diagnostic_accuracy = 0.3
            explanation_parts.append("Correct final diagnosis: +0.3")
        else:
            diagnostic_accuracy = -0.3
            explanation_parts.append("Incorrect final diagnosis: -0.3")

        # Check treatment coverage
        treatments_ok = any(
            any(self._fuzzy_match(rec, appropriate)
                for appropriate in self.appropriate_treatments)
            for rec in self.treatments_recommended
        )

        if treatments_ok:
            treatment_appropriateness = 0.2
            explanation_parts.append("Appropriate treatment plan: +0.2")
        else:
            treatment_appropriateness = -0.1
            explanation_parts.append("Treatment plan issues: -0.1")

        # Check completeness - required tests coverage
        tests_coverage = len(
            self.tests_ordered.intersection(
                set(t.lower() for t in self.required_tests)
            )
        ) / max(len(self.required_tests), 1)

        if tests_coverage >= 0.8:
            efficiency = 0.1
            explanation_parts.append("Good test coverage: +0.1")
        elif tests_coverage >= 0.5:
            efficiency = 0.0
        else:
            efficiency = -0.1
            explanation_parts.append("Insufficient workup: -0.1")

        return {
            "diagnostic_accuracy": diagnostic_accuracy,
            "treatment_appropriateness": treatment_appropriateness,
            "efficiency": efficiency,
            "explanation": " | ".join(explanation_parts)
        }

    def _calculate_progress_bonus(self) -> float:
        """Calculate bonus for making progress toward required actions."""
        # Percentage of required tests ordered
        required_tests_lower = set(t.lower() for t in self.required_tests)
        tests_progress = len(self.tests_ordered.intersection(required_tests_lower))
        tests_total = max(len(required_tests_lower), 1)

        # Percentage of critical questions asked
        critical_q_lower = set(q.lower() for q in self.critical_questions)
        questions_progress = len(self.questions_asked.intersection(critical_q_lower))
        questions_total = max(len(critical_q_lower), 1)

        # Small cumulative bonus for progress
        progress = (tests_progress / tests_total + questions_progress / questions_total) / 2
        return progress * 0.05

    def _fuzzy_match(self, text1: str, text2: str) -> bool:
        """
        Fuzzy matching for clinical terms.
        Handles abbreviations and common variations.
        """
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        # Direct match
        if text1 == text2:
            return True

        # Containment (one contains the other)
        if text1 in text2 or text2 in text1:
            return True

        # Common abbreviation mappings
        abbreviations = {
            "ecg": "electrocardiogram",
            "ekg": "electrocardiogram",
            "cbc": "complete blood count",
            "bmp": "basic metabolic panel",
            "cmp": "comprehensive metabolic panel",
            "lfts": "liver function tests",
            "ua": "urinalysis",
            "cx": "culture",
            "cxr": "chest x-ray",
            "ct": "computed tomography",
            "mri": "magnetic resonance imaging",
            "lp": "lumbar puncture",
            "mi": "myocardial infarction",
            "acs": "acute coronary syndrome",
            "uti": "urinary tract infection",
            "uri": "upper respiratory infection",
            "gbs": "guillain-barre syndrome",
            "hiv": "human immunodeficiency virus",
            "aids": "acquired immunodeficiency syndrome",
            "nstemi": "non-st elevation myocardial infarction",
            "ivig": "intravenous immunoglobulin",
            "ercp": "endoscopic retrograde cholangiopancreatography"
        }

        # Check abbreviation equivalence
        expanded1 = abbreviations.get(text1, text1)
        expanded2 = abbreviations.get(text2, text2)

        if expanded1 == expanded2:
            return True
        if expanded1 in text2 or text2 in expanded1:
            return True
        if expanded2 in text1 or text1 in expanded2:
            return True

        # Word overlap check for longer phrases
        words1 = set(text1.split())
        words2 = set(text2.split())
        if len(words1) > 1 and len(words2) > 1:
            overlap = len(words1.intersection(words2))
            min_words = min(len(words1), len(words2))
            if overlap / min_words >= 0.6:
                return True

        return False

    def get_final_score(self) -> float:
        """
        Calculate final episode score between 0.0 and 1.0.
        Used by task graders.
        """
        score = 0.0

        # Correct diagnosis (40% weight)
        correct_dx_found = any(
            any(self._fuzzy_match(suggested, correct)
                for correct in self.correct_diagnoses)
            for suggested in self.diagnoses_suggested
        )
        if correct_dx_found:
            score += 0.4

        # Required tests coverage (20% weight)
        required_tests_lower = set(t.lower() for t in self.required_tests)
        tests_coverage = len(self.tests_ordered.intersection(required_tests_lower))
        tests_ratio = tests_coverage / max(len(required_tests_lower), 1)
        score += 0.2 * tests_ratio

        # Critical questions coverage (15% weight)
        critical_q_lower = set(q.lower() for q in self.critical_questions)
        questions_coverage = len(self.questions_asked.intersection(critical_q_lower))
        questions_ratio = questions_coverage / max(len(critical_q_lower), 1)
        score += 0.15 * questions_ratio

        # Appropriate treatment (15% weight)
        treatments_ok = any(
            any(self._fuzzy_match(rec, appropriate)
                for appropriate in self.appropriate_treatments)
            for rec in self.treatments_recommended
        )
        if treatments_ok:
            score += 0.15

        # Safety - no contraindicated treatments (10% weight)
        unsafe_treatment = any(
            any(self._fuzzy_match(rec, contra)
                for contra in self.contraindicated_treatments)
            for rec in self.treatments_recommended
        )
        if not unsafe_treatment:
            score += 0.1

        return min(1.0, max(0.0, score))
