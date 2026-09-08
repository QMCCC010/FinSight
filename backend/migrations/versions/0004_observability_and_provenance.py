"""Add collection observability and document provenance.

Revision ID: 0004_observability
Revises: 0003_chat_task_controls
"""

from alembic import op
import sqlalchemy as sa


revision = "0004_observability"
down_revision = "0003_chat_task_controls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("data_sources", sa.Column("last_attempt_at", sa.DateTime(), nullable=True))
    op.add_column("data_sources", sa.Column("last_duration_ms", sa.Integer(), nullable=True))
    op.add_column("data_sources", sa.Column("total_successes", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("data_sources", sa.Column("total_failures", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("data_sources", sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("crawl_runs", sa.Column("source_stats", sa.JSON(), nullable=True))
    op.add_column("documents", sa.Column("acquisition_mode", sa.String(length=16), nullable=False, server_default="LIVE"))
    op.create_index("ix_documents_acquisition_mode", "documents", ["acquisition_mode"], unique=False)
    op.execute("UPDATE documents SET acquisition_mode='SNAPSHOT' WHERE source_url LIKE 'snapshot://%'")


def downgrade() -> None:
    op.drop_index("ix_documents_acquisition_mode", table_name="documents")
    op.drop_column("documents", "acquisition_mode")
    op.drop_column("crawl_runs", "source_stats")
    op.drop_column("data_sources", "consecutive_failures")
    op.drop_column("data_sources", "total_failures")
    op.drop_column("data_sources", "total_successes")
    op.drop_column("data_sources", "last_duration_ms")
    op.drop_column("data_sources", "last_attempt_at")
