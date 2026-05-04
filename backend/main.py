from fastapi import FastAPI

from api.codegen_routes import cursor_cli_router, router as codegen_router
from api.code_assistant_routes import router as code_assistant_router
from api.code_review_routes import router as code_review_router
from api.common_routes import router as common_router
from api.error_fixing_routes import router as error_fixing_router
from api.rca_routes import router as rca_router
from api.test_automation_routes import router as test_automation_router

app = FastAPI(title="Unified Platform API")

app.include_router(common_router, prefix="/api/common", tags=["common"])
app.include_router(codegen_router, prefix="/api/codegen", tags=["codegen"])
app.include_router(cursor_cli_router, tags=["cursor-cli-codegen"])
app.include_router(
    test_automation_router,
    prefix="/api/test-automation",
    tags=["test-automation"],
)
app.include_router(rca_router, prefix="/api/rca", tags=["rca"])
app.include_router(
    code_assistant_router,
    prefix="/api/code-assistant",
    tags=["code-assistant"],
)
app.include_router(
    error_fixing_router,
    prefix="/api/error-fixing",
    tags=["error-fixing"],
)
app.include_router(
    code_review_router,
    prefix="/api/code-review",
    tags=["code-review"],
)
