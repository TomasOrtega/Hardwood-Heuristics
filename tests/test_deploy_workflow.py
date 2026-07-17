from pathlib import Path


WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "deploy.yml"


def test_deploy_requires_tests_and_plot_generation_to_succeed():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "uv run pytest" in content
    assert "continue-on-error" not in content
    assert content.index("uv run pytest") < content.index("uv run python -m src.generate_plots")
    assert content.index("uv run python -m src.generate_plots") < content.index(
        "uv run zensical build"
    )
