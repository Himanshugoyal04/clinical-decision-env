"""
Pydantic models for the Clinical Decision Support Environment.
Defines typed Observation, Action, and Reward models per OpenEnv spec.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of actions the agent can perform."""
    ASK_QUESTION = "ask_question"
    ORDER_TEST = "order_test"
    SUGGEST_DIAGNOSIS = "suggest_diagnosis"
    RECOMMEND_TREATMENT = "recommend_treatment"
    REQUEST_CONSULTATION = "request_consultation"
    FINALIZE_CASE = "finalize_case"


class PatientVitals(BaseModel):
    """Patient vital signs."""
    temperature: Optional[float] = Field(None, description="Body temperature in Celsius")
    blood_pressure_systolic: Optional[int] = Field(None, description="Systolic BP in mmHg")
    blood_pressure_diastolic: Optional[int] = Field(None, description="Diastolic BP in mmHg")
    heart_rate: Optional[int] = Field(None, description="Heart rate in bpm")
    respiratory_rate: Optional[int] = Field(None, description="Respiratory rate per minute")
    oxygen_saturation: Optional[float] = Field(None, description="SpO2 percentage")


class LabResult(BaseModel):
    """Laboratory test result."""
    test_name: str = Field(..., description="Name of the test")
    value: str = Field(..., description="Test result value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    is_abnormal: bool = Field(False, description="Whether result is outside normal range")


class PatientHistory(BaseModel):
    """Patient medical history."""
    age: int = Field(..., description="Patient age in years")
    sex: str = Field(..., description="Patient sex (M/F)")
    chronic_conditions: List[str] = Field(default_factory=list, description="Chronic conditions")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    current_medications: List[str] = Field(default_factory=list, description="Current medications")
    family_history: List[str] = Field(default_factory=list, description="Relevant family history")
    social_history: Optional[str] = Field(None, description="Social history notes")


class Observation(BaseModel):
    """
    Complete observation state visible to the agent.
    Contains all information gathered so far about the patient.
    """
    # Episode metadata
    episode_id: str = Field(..., description="Unique episode identifier")
    step_number: int = Field(0, description="Current step in the episode")
    max_steps: int = Field(20, description="Maximum allowed steps")

    # Initial presentation
    chief_complaint: str = Field(..., description="Patient's main complaint")
    presenting_symptoms: List[str] = Field(default_factory=list, description="Initial symptoms")
    symptom_duration: Optional[str] = Field(None, description="How long symptoms present")

    # Patient information
    patient_history: PatientHistory = Field(..., description="Patient medical history")
    vitals: Optional[PatientVitals] = Field(None, description="Current vital signs")

    # Gathered information (grows as agent acts)
    lab_results: List[LabResult] = Field(default_factory=list, description="Ordered lab results")
    imaging_results: List[str] = Field(default_factory=list, description="Imaging study results")
    question_answers: Dict[str, str] = Field(default_factory=dict, description="Q&A from patient")
    consultation_notes: List[str] = Field(default_factory=list, description="Specialist consultations")

    # Agent's current working state
    current_diagnoses: List[str] = Field(default_factory=list, description="Diagnoses suggested so far")
    current_treatments: List[str] = Field(default_factory=list, description="Treatments recommended")

    # Feedback from last action
    last_action_feedback: Optional[str] = Field(None, description="Feedback from last action")
    last_action_error: Optional[str] = Field(None, description="Error from last action if any")

    # Risk indicators
    urgency_level: str = Field("routine", description="Case urgency: routine/urgent/emergency")
    risk_flags: List[str] = Field(default_factory=list, description="Active risk flags")


class Action(BaseModel):
    """
    Action the agent takes in the clinical environment.
    """
    action_type: ActionType = Field(..., description="Type of action to perform")
    action_value: str = Field(..., description="Value/parameter for the action")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for this action")

    class Config:
        use_enum_values = True


class Reward(BaseModel):
    """
    Reward signal returned after each action.
    Provides detailed breakdown for interpretability.
    """
    total: float = Field(..., description="Total reward for this step", ge=-2.0, le=2.0)

    # Component breakdown
    diagnostic_accuracy: float = Field(0.0, description="Reward for correct diagnostic steps")
    information_gathering: float = Field(0.0, description="Reward for appropriate information gathering")
    treatment_appropriateness: float = Field(0.0, description="Reward for treatment decisions")
    efficiency: float = Field(0.0, description="Reward/penalty for step efficiency")
    safety: float = Field(0.0, description="Penalty for unsafe actions")

    # Explanation
    explanation: str = Field("", description="Human-readable explanation of reward")


class StepResult(BaseModel):
    """Result returned from env.step()."""
    observation: Observation
    reward: Reward
    done: bool = Field(False, description="Whether episode has ended")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info")


class EnvironmentState(BaseModel):
    """Full internal state of the environment (for state() method)."""
    observation: Observation
    ground_truth_diagnosis: List[str] = Field(default_factory=list)
    ground_truth_treatment: List[str] = Field(default_factory=list)
    required_tests: List[str] = Field(default_factory=list)
    critical_questions: List[str] = Field(default_factory=list)
    case_difficulty: str = Field("easy")
    episode_complete: bool = Field(False)
    final_score: Optional[float] = Field(None)
