"""Store crawl-run source selections as a JSON list.

Revision ID: 0008_crawl_run_source_types
Revises: 0007_report_workbench
"""

import json

from alembic import op
import sqlalchemy as sa


revision = "0008_crawl_run_source_types"
down_revision = "0007_report_workbench"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {item["name"] for item in inspector.get_columns("crawl_runs")}
    if "source_types" not in columns:
        # MySQL does not permit a string default on JSON. Add nullable, backfill,
        # then enforce NOT NULL so existing course/demo data is preserved.
        op.add_column("crawl_runs", sa.Column("source_types", sa.JSON(), nullable=True))
        rows = bind.execute(sa.text("SELECT id, source_type FROM crawl_runs")).mappings().all()
        for row in rows:
            values = [item.strip() for item in (row["source_type"] or "").split(",") if item.strip()]
            bind.execute(
                sa.text("UPDATE crawl_runs SET source_types = :source_types WHERE id = :run_id"),
                {"source_types": json.dumps(values, ensure_ascii=False), "run_id": row["id"]},
            )
        op.alter_column("crawl_runs", "source_types", existing_type=sa.JSON(), nullable=False)


def downgrade() -> None:
    columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("crawl_runs")}
    if "source_types" in columns:
        op.drop_column("crawl_runs", "source_types")
