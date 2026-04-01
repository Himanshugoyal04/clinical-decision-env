---
title: Clinical Decision Support Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
---

# Clinical Decision Support Environment

An OpenEnv-compliant AI environment that simulates real-world clinical decision-making scenarios. Agents must analyze patient presentations, gather medical history, order diagnostic tests, and make diagnosis/treatment decisions through multi-step interactions.

## Motivation

Healthcare decision-making is one of the most impactful real-world applications of AI. This environment provides a standardized way to:

- **Train** AI agents on clinical reasoning
- **Evaluate** diagnostic accuracy and treatment appropriateness
- **Benchmark** different AI models on medical decision-making tasks
- **Research** safe AI behavior in high-stakes healthcare scenarios

Unlike toy environments, this simulates the actual workflow of a junior doctor: gathering information, forming differential diagnoses, confirming with tests, and recommending treatments while avoiding dangerous contraindications.

## Features

- 3 difficulty levels with realistic medical cases
- Multi-step diagnostic workflow
- Meaningful partial rewards for progress
- Safety penalties for contraindicated treatments
- Deterministic grading with clear success criteria
- Complete API coverage for reset/step/state

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/your-repo/clinical-decision-env.git
cd clinical-decision-env

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860

# Test the API
curl http://localhost:7860/health
```

### Docker

```bash
# Build
docker build -t clinical-decision-env .

# Run
docker run -p 7860:7860 clinical-decision-env

# Test
curl http://localhost:7860/health
```

### Running Inference

```bash
# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:7860"

# Run baseline inference
python inference.py
```

## Environment API

### Reset

Initialize a new episode with a specific task.

```bash
POST /reset
{
    "task_id": "easy_diagnosis",
    "case_id": null,  # optional: specific case
    "seed": null      # optional: for reproducibility
}
```

Response:
```json
{
    "observation": {...},
    "task_id": "easy_diagnosis",
    "episode_id": "abc123",
    "max_steps": 15
}
```

### Step

Execute an action and receive the next observation.

```bash
POST /step
{
    "action_type": "order_test",
    "action_value": "complete blood count",
    "reasoning": "Check for infection"
}
```

Response:
```json
{
    "observation": {...},
    "reward": {
        "total": 0.2,
        "diagnostic_accuracy": 0.0,
        "information_gathering": 0.2,
        "treatment_appropriateness": 0.0,
        "efficiency": 0.0,
        "safety": 0.0,
        "explanation": "Required test ordered: +0.2"
    },
    "done": false,
    "info": {}
}
```

### State

Get current environment state (including ground truth for debugging).

```bash
GET /state
```

### Health Check

```bash
GET /health
```

## Action Space

| Action Type | Description | Example Value |
|-------------|-------------|---------------|
| `ask_question` | Ask patient a question | "How long have you had fever?" |
| `order_test` | Order diagnostic test | "complete blood count" |
| `suggest_diagnosis` | Propose a diagnosis | "streptococcal pharyngitis" |
| `recommend_treatment` | Recommend treatment | "amoxicillin 500mg" |
| `request_consultation` | Request specialist | "cardiology" |
| `finalize_case` | End episode | "case complete" |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | Unique episode identifier |
| `step_number` | int | Current step (0-indexed) |
| `max_steps` | int | Maximum allowed steps |
| `chief_complaint` | string | Patient's main complaint |
| `presenting_symptoms` | list[str] | Initial symptoms |
| `symptom_duration` | string | How long symptoms present |
| `patient_history` | object | Age, sex, conditions, allergies, medications |
| `vitals` | object | Temperature, BP, HR, RR, SpO2 |
| `lab_results` | list[object] | Ordered test results |
| `question_answers` | dict | Q&A from patient |
| `current_diagnoses` | list[str] | Suggested diagnoses |
| `current_treatments` | list[str] | Recommended treatments |
| `urgency_level` | string | routine/urgent/emergency |
| `risk_flags` | list[str] | Active risk indicators |
| `last_action_feedback` | string | Result of last action |

## Tasks

### Easy: Basic Diagnosis (easy_diagnosis)

**Objective**: Diagnose common conditions with clear presentations.

- Max Steps: 15
- Passing Score: 0.6
- Cases: Strep throat, UTI, Common cold

**Success Criteria**:
- Identify correct diagnosis
- Order relevant tests
- Recommend appropriate treatment
- Avoid contraindicated treatments

### Medium: Complex Diagnosis (medium_diagnosis)

**Objective**: Manage emergency cases requiring multi-step reasoning.

- Max Steps: 20
- Passing Score: 0.55
- Cases: Acute coronary syndrome, Bacterial meningitis, Guillain-Barre syndrome

**Success Criteria**:
- Recognize emergency presentations
- Order comprehensive workup
- Ask critical history questions
- Initiate urgent treatment

### Hard: Expert Diagnosis (hard_diagnosis)

**Objective**: Handle atypical presentations with multiple comorbidities.

- Max Steps: 25
- Passing Score: 0.5
- Cases: Ascending cholangitis, Brugada syndrome, HIV/AIDS

**Success Criteria**:
- Navigate misleading symptoms
- Consider rare diagnoses
- Account for drug interactions
- Recognize critical situations

## Reward Function

The reward function provides signal throughout the episode:

| Component | Weight | Description |
|-----------|--------|-------------|
| Diagnostic Accuracy | 40% | Correct diagnosis identification |
| Test Coverage | 20% | Ordering required diagnostic tests |
| History Gathering | 15% | Asking critical questions |
| Treatment Appropriateness | 15% | Recommending correct treatments |
| Safety | 10% | Avoiding contraindicated treatments |

### Reward Examples

| Action | Reward | Explanation |
|--------|--------|-------------|
| Order required test | +0.2 | Appropriate diagnostic workup |
| Ask critical question | +0.15 | Important history gathering |
| Correct diagnosis | +0.5 | Primary diagnosis match |
| Correct treatment | +0.3 | Appropriate therapy |
| Contraindicated treatment | -1.0 | SAFETY: dangerous action |
| Unnecessary test | -0.05 | Inefficiency |
| Duplicate action | -0.1 | Repetitive behavior |

## Grading

Final scores are calculated at episode end:

```
Score = (0.4 * diagnosis_correct) +
        (0.2 * tests_coverage) +
        (0.15 * questions_coverage) +
        (0.15 * treatment_appropriate) +
        (0.1 * no_unsafe_treatments)
```

Scores range from 0.0 to 1.0.

## Baseline Results

| Task | Score | Steps | Notes |
|------|-------|-------|-------|
| easy_diagnosis | ~0.75 | 8-12 | GPT-4o-mini baseline |
| medium_diagnosis | ~0.60 | 12-18 | Requires multi-step reasoning |
| hard_diagnosis | ~0.45 | 15-22 | Challenges frontier models |

## Project Structure

```
clinical-decision-env/
├── app/
│   ├── __init__.py      # Package exports
│   ├── main.py          # FastAPI server
│   ├── env.py           # Environment class
│   ├── models.py        # Pydantic models
│   ├── tasks.py         # Task definitions & graders
│   ├── rewards.py       # Reward calculation
│   └── cases/
│       └── cases.json   # Medical case data
├── openenv.yaml         # OpenEnv specification
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── inference.py         # Baseline inference script
└── README.md            # This file
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | API key / HF token |
| `ENV_URL` | No | Environment URL (default: localhost:7860) |

## Deployment to Hugging Face Spaces

1. Create a new Space with Docker SDK
2. Upload all files
3. Add tags: `openenv`
4. Configure secrets:
   - `HF_TOKEN`
5. Space will auto-deploy

## Validation

```bash
# Validate OpenEnv compliance
openenv validate .

# Run pre-submission checks
./validate-submission.sh https://your-space.hf.space
```

## Contributing

Contributions welcome! Areas of interest:

- Additional medical cases
- More sophisticated reward shaping
- Clinical accuracy improvements
- New task categories

## License

MIT License

## Acknowledgments

This environment is designed for AI research and education. Medical scenarios are simplified for training purposes and should not be used for actual clinical decision-making.

---

Built with OpenEnv specification compliance for the AI agent evaluation ecosystem.
