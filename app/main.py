"""
FastAPI server for the Clinical Decision Support Environment.
Exposes OpenEnv-compliant REST API endpoints.
"""

from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.env import ClinicalDecisionEnv
from app.models import Action, Observation, StepResult, EnvironmentState
from app.tasks import TaskRegistry


# Request/Response models
class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy_diagnosis"
    case_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str
    action_value: str
    reasoning: Optional[str] = None


class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    passing_score: float


# Create FastAPI app
app = FastAPI(
    title="Clinical Decision Support Environment",
    description="An OpenEnv-compliant environment for training AI agents on clinical decision-making tasks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (per-session in production)
env_instance: Optional[ClinicalDecisionEnv] = None


def get_env() -> ClinicalDecisionEnv:
    """Get or create environment instance."""
    global env_instance
    if env_instance is None:
        env_instance = ClinicalDecisionEnv()
    return env_instance


@app.get("/")
async def root():
    """Root endpoint - basic health check."""
    return {
        "status": "healthy",
        "environment": "clinical-decision-support",
        "version": "1.0.0",
        "openenv_compliant": True
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment validation."""
    return {"status": "ok"}


@app.post("/reset", response_model=Dict[str, Any])
async def reset(request: ResetRequest):
    """
    Reset the environment to initial state.

    Returns initial observation for the specified task.
    """
    global env_instance

    try:
        env_instance = ClinicalDecisionEnv(
            task_id=request.task_id or "easy_diagnosis",
            case_id=request.case_id,
            seed=request.seed
        )

        observation = env_instance.reset()

        return {
            "observation": observation.model_dump(),
            "task_id": env_instance.task_id,
            "episode_id": env_instance.episode_id,
            "max_steps": observation.max_steps
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=Dict[str, Any])
async def step(request: StepRequest):
    """
    Execute a single step in the environment.

    Takes an action and returns observation, reward, done, info.
    """
    env = get_env()

    if env.observation is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    try:
        action = Action(
            action_type=request.action_type,
            action_value=request.action_value,
            reasoning=request.reasoning
        )

        result = env.step(action)

        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward.model_dump(),
            "done": result.done,
            "info": result.info
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=Dict[str, Any])
async def state():
    """
    Get current environment state.

    Returns full state including ground truth for debugging.
    """
    env = get_env()

    if env.observation is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    try:
        env_state = env.state()
        return env_state.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks", response_model=Dict[str, Any])
async def list_tasks():
    """
    List all available tasks.

    Returns task definitions with difficulty levels and requirements.
    """
    registry = TaskRegistry()
    return {
        "tasks": registry.list_tasks(),
        "total_tasks": len(registry.tasks)
    }


@app.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: str):
    """
    Get details for a specific task.
    """
    registry = TaskRegistry()
    task = registry.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return {
        "task_id": task.task_id,
        "name": task.name,
        "description": task.description,
        "difficulty": task.difficulty,
        "max_steps": task.max_steps,
        "passing_score": task.passing_score,
        "objective": task.objective,
        "success_criteria": task.success_criteria,
        "case_ids": task.case_ids
    }


@app.get("/action_space", response_model=Dict[str, Any])
async def action_space():
    """
    Get description of the action space.
    """
    env = get_env()
    return env.get_action_space()


@app.get("/observation_space", response_model=Dict[str, Any])
async def observation_space():
    """
    Get description of the observation space.
    """
    env = get_env()
    return env.get_observation_space()


@app.post("/close")
async def close():
    """
    Close the current environment instance.
    """
    global env_instance
    if env_instance:
        env_instance.close()
        env_instance = None
    return {"status": "closed"}


@app.get("/grade", response_model=Dict[str, Any])
async def grade():
    """
    Get grading results for the current episode.

    Only available after episode is complete.
    """
    env = get_env()

    if env.observation is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    if not env.episode_done:
        raise HTTPException(
            status_code=400,
            detail="Episode not complete. Finalize the case first."
        )

    try:
        if env.grader is None:
            raise HTTPException(status_code=500, detail="Grader not initialized")

        grade_result = env.grader.grade_episode(
            actions_taken=env.actions_taken,
            final_diagnoses=env.diagnoses_suggested,
            final_treatments=env.treatments_recommended,
            tests_ordered=env.tests_ordered,
            questions_asked=env.questions_asked,
            steps_used=env.step_count
        )

        return grade_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Entry point for direct execution
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )
