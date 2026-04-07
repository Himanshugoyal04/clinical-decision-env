#!/usr/bin/env python3
"""
Baseline Inference Script for Clinical Decision Support Environment.

This script runs an LLM agent against the environment across all 3 task
difficulty levels and produces reproducible baseline scores.

Required Environment Variables:
    - API_BASE_URL: The API endpoint for the LLM
    - MODEL_NAME: The model identifier to use for inference
    - HF_TOKEN: Your Hugging Face / API key (used as API key)

Usage:
    python inference.py [--env-url ENV_URL] [--debug]
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, List, Optional

import requests
from openai import OpenAI


# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")  # No default - must be provided
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

MAX_STEPS = 20
MAX_TOKENS = 1024
TEMPERATURE = 0.2
DEBUG = False

# System prompt for the clinical decision agent
SYSTEM_PROMPT = """You are an AI clinical decision support assistant acting as a junior doctor.
Your goal is to diagnose the patient's condition and recommend appropriate treatment.

You have access to the following actions:
1. ask_question - Ask the patient a question to gather more history
2. order_test - Order a diagnostic test (e.g., "complete blood count", "ECG", "chest x-ray")
3. suggest_diagnosis - Suggest a diagnosis based on your findings
4. recommend_treatment - Recommend a treatment for the condition
5. request_consultation - Request a specialist consultation
6. finalize_case - Finalize the case when you have reached your conclusions

IMPORTANT GUIDELINES:
- Gather relevant history through targeted questions
- Order appropriate diagnostic tests to confirm your suspicions
- Consider the patient's allergies and contraindications
- Make a clear diagnosis before recommending treatment
- Finalize the case when you have completed your assessment

Respond with EXACTLY ONE action in JSON format:
{
    "action_type": "<action_type>",
    "action_value": "<value>",
    "reasoning": "<brief explanation>"
}

Do not include any other text outside the JSON object."""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline inference on Clinical Decision Environment"
    )
    parser.add_argument(
        "--env-url",
        type=str,
        default=ENV_URL,
        help="URL of the environment server"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["easy_diagnosis", "medium_diagnosis", "hard_diagnosis"],
        help="Tasks to run inference on"
    )
    return parser.parse_args()


def create_client() -> OpenAI:
    """Create OpenAI client with configured API settings."""
    return OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL
    )


def build_user_prompt(
    step: int,
    observation: Dict[str, Any],
    history: List[str]
) -> str:
    """Build the user prompt from observation and history."""
    obs = observation

    prompt_parts = [
        f"=== Clinical Case - Step {step}/{obs.get('max_steps', 20)} ===\n",
        f"Chief Complaint: {obs.get('chief_complaint', 'Unknown')}",
        f"Presenting Symptoms: {', '.join(obs.get('presenting_symptoms', []))}",
        f"Duration: {obs.get('symptom_duration', 'Unknown')}",
        f"Urgency: {obs.get('urgency_level', 'routine').upper()}",
        "\n--- Patient Information ---"
    ]

    # Patient history
    patient_hist = obs.get("patient_history", {})
    prompt_parts.append(f"Age: {patient_hist.get('age', 'Unknown')} | Sex: {patient_hist.get('sex', 'Unknown')}")

    if patient_hist.get("chronic_conditions"):
        prompt_parts.append(f"Chronic Conditions: {', '.join(patient_hist['chronic_conditions'])}")

    if patient_hist.get("allergies"):
        prompt_parts.append(f"ALLERGIES: {', '.join(patient_hist['allergies'])}")

    if patient_hist.get("current_medications"):
        prompt_parts.append(f"Current Medications: {', '.join(patient_hist['current_medications'])}")

    if patient_hist.get("family_history"):
        prompt_parts.append(f"Family History: {', '.join(patient_hist['family_history'])}")

    # Vitals
    vitals = obs.get("vitals")
    if vitals:
        prompt_parts.append("\n--- Vital Signs ---")
        if vitals.get("temperature"):
            prompt_parts.append(f"Temperature: {vitals['temperature']}C")
        if vitals.get("blood_pressure_systolic"):
            prompt_parts.append(
                f"Blood Pressure: {vitals['blood_pressure_systolic']}/{vitals.get('blood_pressure_diastolic', '?')} mmHg"
            )
        if vitals.get("heart_rate"):
            prompt_parts.append(f"Heart Rate: {vitals['heart_rate']} bpm")
        if vitals.get("respiratory_rate"):
            prompt_parts.append(f"Respiratory Rate: {vitals['respiratory_rate']}/min")
        if vitals.get("oxygen_saturation"):
            prompt_parts.append(f"SpO2: {vitals['oxygen_saturation']}%")

    # Lab results gathered so far
    lab_results = obs.get("lab_results", [])
    if lab_results:
        prompt_parts.append("\n--- Test Results ---")
        for result in lab_results:
            abnormal = " [ABNORMAL]" if result.get("is_abnormal") else ""
            prompt_parts.append(f"- {result.get('test_name', 'Unknown')}: {result.get('value', 'N/A')}{abnormal}")

    # Questions and answers
    qa = obs.get("question_answers", {})
    if qa:
        prompt_parts.append("\n--- Patient Responses ---")
        for q, a in qa.items():
            prompt_parts.append(f"Q: {q}\nA: {a}")

    # Current working diagnoses and treatments
    diagnoses = obs.get("current_diagnoses", [])
    treatments = obs.get("current_treatments", [])
    if diagnoses:
        prompt_parts.append(f"\n--- Your Current Diagnoses: {', '.join(diagnoses)} ---")
    if treatments:
        prompt_parts.append(f"--- Your Recommended Treatments: {', '.join(treatments)} ---")

    # Last action feedback
    feedback = obs.get("last_action_feedback")
    if feedback:
        prompt_parts.append(f"\n[Last Action Result: {feedback}]")

    # Risk flags
    risk_flags = obs.get("risk_flags", [])
    if risk_flags:
        prompt_parts.append(f"\n[RISK FLAGS: {', '.join(risk_flags)}]")

    # Action history
    if history:
        prompt_parts.append("\n--- Action History ---")
        for h in history[-5:]:  # Last 5 actions
            prompt_parts.append(h)

    prompt_parts.append("\n\nWhat is your next action? Respond with JSON only.")

    return "\n".join(prompt_parts)


def parse_model_action(response_text: str) -> Dict[str, str]:
    """Parse the model's response to extract action."""
    # Try to extract JSON from response
    try:
        # Find JSON in response
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            action = json.loads(json_str)

            return {
                "action_type": action.get("action_type", "finalize_case"),
                "action_value": action.get("action_value", "case complete"),
                "reasoning": action.get("reasoning", "")
            }
    except json.JSONDecodeError:
        pass

    # Fallback: try to parse simple format
    if "ask_question" in response_text.lower():
        return {"action_type": "ask_question", "action_value": "What other symptoms do you have?", "reasoning": ""}
    if "order_test" in response_text.lower():
        return {"action_type": "order_test", "action_value": "complete blood count", "reasoning": ""}
    if "suggest_diagnosis" in response_text.lower():
        return {"action_type": "suggest_diagnosis", "action_value": "viral infection", "reasoning": ""}
    if "recommend_treatment" in response_text.lower():
        return {"action_type": "recommend_treatment", "action_value": "symptomatic treatment", "reasoning": ""}

    # Default fallback
    return {"action_type": "finalize_case", "action_value": "case complete", "reasoning": "parsing failed"}


def log_step(step: int, action: Dict, reward: float, done: bool, error: Optional[str]):
    """Log step information in required structured format."""
    # Print structured output for validator
    print(f"[STEP] step={step} reward={reward}", flush=True)
    
    # Additional debug info
    action_str = f"{action['action_type']}: {action['action_value'][:50]}..."
    status = "DONE" if done else "OK"
    error_str = f" ERROR: {error}" if error else ""
    print(f"  Step {step:2d} | {action_str:60s} | R: {reward:+.3f} | {status}{error_str}", flush=True)


def log_start(task_id: str, case_id: str):
    """Log episode start in required structured format."""
    # Print structured output for validator
    print(f"[START] task={task_id}", flush=True)
    
    # Additional debug info
    print(f"\n{'='*70}", flush=True)
    print(f"Starting Episode: Task={task_id}, Case={case_id}", flush=True)
    print("="*70, flush=True)


def log_end(task_id: str, success: bool, steps: int, rewards: List[float], grade: Optional[Dict]):
    """Log episode end in required structured format."""
    total_reward = sum(rewards)
    avg_reward = total_reward / max(len(rewards), 1)
    score = grade.get("score", 0.0) if grade else 0.0
    
    # Print structured output for validator
    print(f"[END] task={task_id} score={score} steps={steps}", flush=True)

    # Additional debug info
    print("-"*70, flush=True)
    print(f"Episode Complete | Steps: {steps} | Total Reward: {total_reward:+.3f} | Avg: {avg_reward:+.3f}", flush=True)

    if grade:
        print(f"Final Score: {grade.get('score', 0):.3f} | Passed: {grade.get('passed', False)}", flush=True)
        if grade.get("feedback"):
            print("Feedback:", flush=True)
            for fb in grade.get("feedback", []):
                print(f"  - {fb}", flush=True)

    print("="*70 + "\n", flush=True)


def run_episode(
    client: OpenAI,
    env_url: str,
    task_id: str,
    debug: bool = False
) -> Dict[str, Any]:
    """Run a single episode on the environment."""
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    success = False
    final_grade = None

    try:
        # Reset environment
        reset_response = requests.post(
            f"{env_url}/reset",
            json={"task_id": task_id}
        )
        reset_response.raise_for_status()
        reset_data = reset_response.json()

        observation = reset_data["observation"]
        episode_id = reset_data.get("episode_id", "unknown")
        max_steps = observation.get("max_steps", MAX_STEPS)

        log_start(task_id, episode_id)

        done = False

        for step in range(1, max_steps + 1):
            if done:
                break

            # Build prompt
            user_prompt = build_user_prompt(step, observation, history)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            # Get model response
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                if debug:
                    print(f"[DEBUG] Model request failed: {exc}")
                response_text = '{"action_type": "finalize_case", "action_value": "case complete"}'

            # Parse action
            action = parse_model_action(response_text)

            if debug:
                print(f"[DEBUG] Model response: {response_text[:200]}...")
                print(f"[DEBUG] Parsed action: {action}")

            # Execute step
            step_response = requests.post(
                f"{env_url}/step",
                json=action
            )
            step_response.raise_for_status()
            step_data = step_response.json()

            observation = step_data["observation"]
            reward_data = step_data.get("reward", {})
            reward = reward_data.get("total", 0.0)
            done = step_data.get("done", False)
            info = step_data.get("info", {})
            error = observation.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error)

            # Update history
            history_line = f"Step {step}: {action['action_type']}({action['action_value'][:30]}) -> R:{reward:+.2f}"
            history.append(history_line)

            if done:
                final_grade = info.get("final_grade")
                success = final_grade.get("passed", False) if final_grade else False
                break

        # Get final grade if not already obtained
        if done and not final_grade:
            try:
                grade_response = requests.get(f"{env_url}/grade")
                if grade_response.status_code == 200:
                    final_grade = grade_response.json()
            except Exception:
                pass

        log_end(task_id, success, steps_taken, rewards, final_grade)

    except requests.RequestException as e:
        print(f"[ERROR] Request failed: {e}", flush=True)
        # Print END even on error so validator sees structured output
        print(f"[END] task={task_id} score=0.0 steps={steps_taken}", flush=True)
        final_grade = {"score": 0.0, "passed": False, "error": str(e)}

    finally:
        # Close environment
        try:
            requests.post(f"{env_url}/close")
        except Exception:
            pass

    return {
        "task_id": task_id,
        "steps": steps_taken,
        "total_reward": sum(rewards),
        "rewards": rewards,
        "success": success,
        "grade": final_grade
    }


def main():
    """Main entry point."""
    args = parse_args()
    global DEBUG
    DEBUG = args.debug

    # Validate environment variables
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN or OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    print("="*70)
    print("Clinical Decision Support Environment - Baseline Inference")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Environment: {args.env_url}")
    print(f"Tasks: {args.tasks}")
    print("="*70)

    # Create OpenAI client
    client = create_client()

    # Run inference on all tasks
    results = []
    for task_id in args.tasks:
        result = run_episode(
            client=client,
            env_url=args.env_url,
            task_id=task_id,
            debug=args.debug
        )
        results.append(result)

        # Brief pause between episodes
        time.sleep(1)

    # Print summary
    print("\n" + "="*70)
    print("BASELINE RESULTS SUMMARY")
    print("="*70)
    print(f"{'Task':<25} {'Score':>8} {'Passed':>8} {'Steps':>8} {'Reward':>10}")
    print("-"*70)

    total_score = 0.0
    passed_count = 0

    for r in results:
        grade = r.get("grade", {})
        score = grade.get("score", 0.0)
        passed = "Yes" if grade.get("passed", False) else "No"
        steps = r.get("steps", 0)
        reward = r.get("total_reward", 0.0)

        print(f"{r['task_id']:<25} {score:>8.3f} {passed:>8} {steps:>8} {reward:>+10.3f}")

        total_score += score
        if grade.get("passed", False):
            passed_count += 1

    print("-"*70)
    avg_score = total_score / max(len(results), 1)
    print(f"{'Average':<25} {avg_score:>8.3f} {passed_count}/{len(results):>6}")
    print("="*70)

    # Save results to file
    output_file = "baseline_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "api_base": API_BASE_URL,
            "results": results,
            "summary": {
                "average_score": avg_score,
                "passed_count": passed_count,
                "total_tasks": len(results)
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
